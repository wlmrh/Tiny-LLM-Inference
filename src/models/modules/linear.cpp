#include "tiny_llm/models/modules/linear.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/operators/matmul.h"

#include <array>
#include <stdexcept>
#include <string>

namespace tiny_llm {
namespace modules {

namespace {

Tensor make_2d_f32_tensor_view(void* data, int64_t rows, int64_t cols)
{
    const auto options = torch::TensorOptions()
        .dtype(to_torch_scalar_type(DType::kFloat32))
        .device(infer_blob_device(data));
    const std::array<int64_t, 2> shape{rows, cols};
    return torch::from_blob(data, shape, options);
}

Tensor make_weight_tensor(const StackedWeightDesc& desc)
{
    if (desc.weight.defined())
    {
        return desc.weight;
    }

    if (desc.layout == WeightLayout::kInOut)
    {
        return make_2d_f32_tensor_view(
            desc.data,
            static_cast<int64_t>(desc.in_features),
            static_cast<int64_t>(desc.out_features));
    }

    return make_2d_f32_tensor_view(
        desc.data,
        static_cast<int64_t>(desc.out_features),
        static_cast<int64_t>(desc.in_features));
}

bool needs_torch_matmul_path(const Tensor& input, const Tensor& output, const Tensor& weight)
{
    return input.device().is_cuda() || output.device().is_cuda() || weight.device().is_cuda();
}

void add_bias_to_output_slice(Tensor& output,
                              int32_t output_offset,
                              int32_t out_features,
                              const Tensor& bias)
{
    if (!bias.defined())
    {
        return;
    }
    if (tensor_dtype(bias) != DType::kFloat32 || bias.dim() != 1 || bias.size(0) != out_features)
    {
        throw std::runtime_error("modules::Linear::forward: bias tensor shape/dtype mismatch.");
    }
    if (output.device() != bias.device())
    {
        throw std::runtime_error("modules::Linear::forward: output and bias devices must match.");
    }

    output.narrow(1, output_offset, out_features).add_(bias);
}

void run_out_in_matmul(const Tensor& input,
                       const Tensor& weight_out_in,
                       Tensor& output)
{
    if (input.device() != weight_out_in.device() || input.device() != output.device())
    {
        throw std::runtime_error("modules::Linear::forward: input, weight, and output devices must match.");
    }
    const Tensor projected = torch::matmul(input, weight_out_in.transpose(0, 1));
    output.copy_(projected);
}

float compute_linear_sum(const float* input_ptr,
                         const float* weight_ptr,
                         int32_t in_features,
                         int32_t out_features,
                         int32_t out_col,
                         WeightLayout layout)
{
    float sum = 0.0f;
    for (int32_t k = 0; k < in_features; ++k)
    {
        if (layout == WeightLayout::kInOut)
        {
            sum += input_ptr[static_cast<size_t>(k)]
                * weight_ptr[static_cast<size_t>(k) * static_cast<size_t>(out_features)
                             + static_cast<size_t>(out_col)];
        }
        else
        {
            sum += input_ptr[static_cast<size_t>(k)]
                * weight_ptr[static_cast<size_t>(out_col) * static_cast<size_t>(in_features)
                             + static_cast<size_t>(k)];
        }
    }
    return sum;
}

} // namespace

Linear::Linear(int32_t in_features, int32_t out_features_total)
    : in_features_(in_features), out_features_total_(out_features_total)
{
    if (in_features_ <= 0)
    {
        throw std::runtime_error("modules::Linear: in_features must be positive.");
    }
    if (out_features_total_ <= 0)
    {
        throw std::runtime_error("modules::Linear: out_features_total must be positive.");
    }
}

void Linear::bind_weight(const Tensor& weight, WeightLayout layout)
{
    if (!weight.defined())
    {
        throw std::runtime_error("modules::Linear::bind_weight: weight tensor must be defined.");
    }
    if (tensor_dtype(weight) != DType::kFloat32)
    {
        throw std::runtime_error("modules::Linear::bind_weight: weight tensor must be float32.");
    }
    if (weight.dim() != 2)
    {
        throw std::runtime_error("modules::Linear::bind_weight: weight tensor must be rank-2.");
    }

    const int64_t expected_rows = layout == WeightLayout::kInOut ? in_features_ : out_features_total_;
    const int64_t expected_cols = layout == WeightLayout::kInOut ? out_features_total_ : in_features_;
    if (weight.size(0) != expected_rows || weight.size(1) != expected_cols)
    {
        throw std::runtime_error("modules::Linear::bind_weight: weight tensor shape mismatches module configuration.");
    }
    if (tensor_data(weight) == nullptr)
    {
        throw std::runtime_error("modules::Linear::bind_weight: weight tensor data pointer must be non-null.");
    }

    single_weight_ = StackedWeightDesc{
        static_cast<float*>(tensor_data(weight)),
        out_features_total_,
        in_features_,
        0,
        layout,
        register_parameter("weight", weight, /*requires_grad=*/false),
        Tensor{},
    };
    stacked_weights_.clear();
    stacked_weight_cache_ = Tensor{};
    stacked_bias_cache_ = Tensor{};
    use_stacked_weights_ = false;
}

void Linear::bind_weight(float* weight,
                         int32_t out_features,
                         int32_t in_features,
                         WeightLayout layout)
{
    if (weight == nullptr)
    {
        throw std::runtime_error("modules::Linear::bind_weight: weight pointer must be non-null.");
    }
    if (in_features != in_features_)
    {
        throw std::runtime_error("modules::Linear::bind_weight: in_features mismatches module configuration.");
    }
    if (out_features != out_features_total_)
    {
        throw std::runtime_error("modules::Linear::bind_weight: out_features mismatches module configuration.");
    }

    Tensor weight_tensor = layout == WeightLayout::kInOut
        ? make_2d_f32_tensor_view(weight, in_features, out_features)
        : make_2d_f32_tensor_view(weight, out_features, in_features);
    single_weight_ = StackedWeightDesc{
        weight,
        out_features,
        in_features,
        0,
        layout,
        register_parameter("weight", weight_tensor, /*requires_grad=*/false),
        Tensor{},
    };
    stacked_weights_.clear();
    stacked_weight_cache_ = Tensor{};
    stacked_bias_cache_ = Tensor{};
    use_stacked_weights_ = false;
}

void Linear::bind_stacked_weights(const StackedWeightDesc* descs, int32_t count)
{
    validate_descs(descs, count);
    stacked_weights_.clear();
    stacked_weights_.reserve(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i)
    {
        StackedWeightDesc desc = descs[i];
        if (!desc.weight.defined())
        {
            desc.weight = make_weight_tensor(desc);
        }
        desc.weight = register_parameter(
            "weight_" + std::to_string(i),
            desc.weight,
            /*requires_grad=*/false);
        if (desc.bias.defined())
        {
            desc.bias = register_parameter(
                "bias_" + std::to_string(i),
                desc.bias,
                /*requires_grad=*/false);
        }
        desc.data = static_cast<float*>(tensor_data(desc.weight));
        stacked_weights_.push_back(std::move(desc));
    }
    use_stacked_weights_ = true;
    build_stacked_weight_cache();
}

Tensor Linear::forward(const Tensor& input, ExecutionContext& ctx) const
{
    if (!input.defined())
    {
        throw std::runtime_error("modules::Linear::forward: input must be defined.");
    }
    if (input.dim() != 2)
    {
        throw std::runtime_error("modules::Linear::forward: input must be rank-2.");
    }

    Tensor output = torch::empty(
        {input.size(0), out_features_total_},
        torch::TensorOptions().dtype(to_torch_scalar_type(DType::kFloat32)).device(input.device()));
    forward(input, output, ctx);
    return output;
}

void Linear::forward(const Tensor& input, Tensor& output, ExecutionContext& ctx) const
{
    validate_forward_inputs(input, output);

    const int64_t rows = input.size(0);
    if (!use_stacked_weights_)
    {
        if (single_weight_.layout == WeightLayout::kInOut)
        {
            Tensor weight = make_weight_tensor(single_weight_);
            ops::gemm(input, weight, output, ctx);
            return;
        }

        Tensor weight = make_weight_tensor(single_weight_);
        if (needs_torch_matmul_path(input, output, weight))
        {
            run_out_in_matmul(input, weight, output);
            return;
        }

        const float* input_ptr = static_cast<const float*>(tensor_data(input));
        const float* weight_ptr = static_cast<const float*>(tensor_data(weight));
        float* output_ptr = static_cast<float*>(tensor_data(output));
        for (int64_t row = 0; row < rows; ++row)
        {
            const float* input_row_ptr =
                input_ptr + static_cast<size_t>(row) * static_cast<size_t>(in_features_);
            float* output_row_ptr =
                output_ptr + static_cast<size_t>(row) * static_cast<size_t>(out_features_total_);
            for (int32_t out_col = 0; out_col < out_features_total_; ++out_col)
            {
                output_row_ptr[static_cast<size_t>(out_col)] = compute_linear_sum(
                    input_row_ptr,
                    weight_ptr,
                    in_features_,
                    out_features_total_,
                    out_col,
                    single_weight_.layout);
            }
        }
        return;
    }

    if (stacked_weight_cache_.defined()
        && needs_torch_matmul_path(input, output, stacked_weight_cache_))
    {
        if (stacked_weight_cache_layout_ == WeightLayout::kInOut)
        {
            ops::gemm(input, stacked_weight_cache_, output, ctx);
        }
        else
        {
            run_out_in_matmul(input, stacked_weight_cache_, output);
        }
        if (stacked_bias_cache_.defined())
        {
            if (output.device() != stacked_bias_cache_.device())
            {
                throw std::runtime_error("modules::Linear::forward: output and stacked bias devices must match.");
            }
            output.add_(stacked_bias_cache_);
        }
        return;
    }

    for (const StackedWeightDesc& desc : stacked_weights_)
    {
        Tensor weight = make_weight_tensor(desc);

        if (needs_torch_matmul_path(input, output, weight))
        {
            Tensor output_slice = output.narrow(1, desc.output_offset, desc.out_features);
            if (desc.layout == WeightLayout::kInOut)
            {
                if (input.device() != weight.device() || input.device() != output.device())
                {
                    throw std::runtime_error("modules::Linear::forward: input, weight, and output devices must match.");
                }
                output_slice.copy_(torch::matmul(input, weight));
            }
            else
            {
                run_out_in_matmul(input, weight, output_slice);
            }
            add_bias_to_output_slice(output, desc.output_offset, desc.out_features, desc.bias);
            continue;
        }

        const float* weight_ptr = static_cast<const float*>(tensor_data(weight));
        const float* input_ptr = static_cast<const float*>(tensor_data(input));
        float* output_ptr = static_cast<float*>(tensor_data(output));

        for (int64_t row = 0; row < rows; ++row)
        {
            const size_t input_row_offset = static_cast<size_t>(row) * static_cast<size_t>(in_features_);
            const size_t output_row_offset = static_cast<size_t>(row) * static_cast<size_t>(out_features_total_);
            for (int32_t out_col = 0; out_col < desc.out_features; ++out_col)
            {
                const float sum = compute_linear_sum(
                    input_ptr + input_row_offset,
                    weight_ptr,
                    in_features_,
                    desc.out_features,
                    out_col,
                    desc.layout);
                output_ptr[output_row_offset + static_cast<size_t>(desc.output_offset + out_col)] = sum;
            }
        }
        if (desc.bias.defined())
        {
            const float* bias_ptr = static_cast<const float*>(tensor_data(desc.bias));
            if (bias_ptr == nullptr)
            {
                throw std::runtime_error("modules::Linear::forward: bias data pointer must be non-null.");
            }
            for (int64_t row = 0; row < rows; ++row)
            {
                const size_t output_row_offset =
                    static_cast<size_t>(row) * static_cast<size_t>(out_features_total_);
                for (int32_t out_col = 0; out_col < desc.out_features; ++out_col)
                {
                    output_ptr[output_row_offset + static_cast<size_t>(desc.output_offset + out_col)] +=
                        bias_ptr[static_cast<size_t>(out_col)];
                }
            }
        }
    }
}

void Linear::validate_forward_inputs(const Tensor& input, const Tensor& output) const
{
    if (!input.defined() || !output.defined())
    {
        throw std::runtime_error("modules::Linear::forward: input and output must be defined.");
    }
    if (tensor_dtype(input) != DType::kFloat32 || tensor_dtype(output) != DType::kFloat32)
    {
        throw std::runtime_error("modules::Linear::forward: only float32 tensors are supported.");
    }
    if (input.dim() != 2 || output.dim() != 2)
    {
        throw std::runtime_error("modules::Linear::forward: input and output must be rank-2 tensors.");
    }
    if (input.size(1) != in_features_)
    {
        throw std::runtime_error("modules::Linear::forward: input last dimension mismatches in_features.");
    }
    if (output.size(0) != input.size(0) || output.size(1) != out_features_total_)
    {
        throw std::runtime_error("modules::Linear::forward: output shape must be [tokens, out_features_total].");
    }
    if (tensor_data(input) == nullptr || tensor_data(output) == nullptr)
    {
        throw std::runtime_error("modules::Linear::forward: input/output pointers must be non-null.");
    }
    if (!use_stacked_weights_ && single_weight_.data == nullptr)
    {
        throw std::runtime_error("modules::Linear::forward: no weight is bound.");
    }
    if (use_stacked_weights_ && stacked_weights_.empty())
    {
        throw std::runtime_error("modules::Linear::forward: stacked weights are not bound.");
    }
}

void Linear::build_stacked_weight_cache()
{
    stacked_weight_cache_ = Tensor{};
    stacked_bias_cache_ = Tensor{};
    if (stacked_weights_.empty())
    {
        return;
    }

    const WeightLayout layout = stacked_weights_.front().layout;
    std::vector<Tensor> weights;
    weights.reserve(stacked_weights_.size());
    std::vector<Tensor> biases;
    biases.reserve(stacked_weights_.size());
    bool has_any_bias = false;
    for (const StackedWeightDesc& desc : stacked_weights_)
    {
        if (desc.layout != layout || !desc.weight.defined())
        {
            return;
        }
        weights.push_back(desc.weight);
        if (desc.bias.defined())
        {
            has_any_bias = true;
            biases.push_back(desc.bias);
        }
        else
        {
            biases.push_back(torch::zeros(
                {desc.out_features},
                torch::TensorOptions().dtype(torch::kFloat32).device(desc.weight.device())));
        }
    }

    const int64_t cat_dim = layout == WeightLayout::kInOut ? 1 : 0;
    stacked_weight_cache_ = register_parameter(
        "stacked_weight_cache",
        torch::cat(weights, cat_dim).contiguous(),
        /*requires_grad=*/false);
    stacked_weight_cache_layout_ = layout;
    if (has_any_bias)
    {
        stacked_bias_cache_ = register_parameter(
            "stacked_bias_cache",
            torch::cat(biases, 0).contiguous(),
            /*requires_grad=*/false);
    }
}

void Linear::validate_descs(const StackedWeightDesc* descs, int32_t count) const
{
    if (descs == nullptr)
    {
        throw std::runtime_error("modules::Linear::bind_stacked_weights: descs must be non-null.");
    }
    if (count <= 0)
    {
        throw std::runtime_error("modules::Linear::bind_stacked_weights: count must be positive.");
    }

    int32_t covered_out_features = 0;
    for (int32_t i = 0; i < count; ++i)
    {
        const StackedWeightDesc& desc = descs[i];
        if (desc.weight.defined())
        {
            if (tensor_dtype(desc.weight) != DType::kFloat32)
            {
                throw std::runtime_error("modules::Linear::bind_stacked_weights: desc.weight must be float32.");
            }
            if (desc.weight.dim() != 2)
            {
                throw std::runtime_error("modules::Linear::bind_stacked_weights: desc.weight must be rank-2.");
            }
            const int64_t expected_rows = desc.layout == WeightLayout::kInOut
                ? desc.in_features
                : desc.out_features;
            const int64_t expected_cols = desc.layout == WeightLayout::kInOut
                ? desc.out_features
                : desc.in_features;
            if (desc.weight.size(0) != expected_rows || desc.weight.size(1) != expected_cols)
            {
                throw std::runtime_error("modules::Linear::bind_stacked_weights: desc.weight shape mismatch.");
            }
            if (tensor_data(desc.weight) == nullptr)
            {
                throw std::runtime_error("modules::Linear::bind_stacked_weights: desc.weight data pointer must be non-null.");
            }
        }
        else if (desc.data == nullptr)
        {
            throw std::runtime_error("modules::Linear::bind_stacked_weights: desc weight must be bound.");
        }
        if (desc.bias.defined())
        {
            if (tensor_dtype(desc.bias) != DType::kFloat32
                || desc.bias.dim() != 1
                || desc.bias.size(0) != desc.out_features)
            {
                throw std::runtime_error("modules::Linear::bind_stacked_weights: desc.bias shape/dtype mismatch.");
            }
            if (tensor_data(desc.bias) == nullptr)
            {
                throw std::runtime_error("modules::Linear::bind_stacked_weights: desc.bias data pointer must be non-null.");
            }
        }
        if (desc.in_features != in_features_)
        {
            throw std::runtime_error("modules::Linear::bind_stacked_weights: desc.in_features mismatches module configuration.");
        }
        if (desc.out_features <= 0)
        {
            throw std::runtime_error("modules::Linear::bind_stacked_weights: desc.out_features must be positive.");
        }
        if (desc.output_offset != covered_out_features)
        {
            throw std::runtime_error("modules::Linear::bind_stacked_weights: output offsets must be contiguous and sorted.");
        }

        covered_out_features += desc.out_features;
        if (covered_out_features > out_features_total_)
        {
            throw std::runtime_error("modules::Linear::bind_stacked_weights: stacked weights exceed output dimension.");
        }
    }

    if (covered_out_features != out_features_total_)
    {
        throw std::runtime_error("modules::Linear::bind_stacked_weights: stacked weights must fully cover output dimension.");
    }
}

} // namespace modules
} // namespace tiny_llm
