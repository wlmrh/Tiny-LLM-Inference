#include "tiny_llm/models/modules/linear.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/operators/matmul.h"

#include <array>
#include <stdexcept>

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

bool needs_torch_matmul_path(const Tensor& input, const Tensor& output, const Tensor& weight)
{
    return input.device().is_cuda() || output.device().is_cuda() || weight.device().is_cuda();
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

    single_weight_ = StackedWeightDesc{weight, out_features, in_features, 0, layout};
    stacked_weights_ = nullptr;
    stacked_weight_count_ = 0;
    use_stacked_weights_ = false;
}

void Linear::bind_stacked_weights(const StackedWeightDesc* descs, int32_t count)
{
    validate_descs(descs, count);
    stacked_weights_ = descs;
    stacked_weight_count_ = count;
    use_stacked_weights_ = true;
}

void Linear::forward(const Tensor& input, Tensor& output, ExecutionContext& ctx) const
{
    validate_forward_inputs(input, output);

    const int64_t rows = input.size(0);
    if (!use_stacked_weights_)
    {
        if (single_weight_.layout == WeightLayout::kInOut)
        {
            Tensor weight = make_2d_f32_tensor_view(
                single_weight_.data,
                static_cast<int64_t>(single_weight_.in_features),
                static_cast<int64_t>(single_weight_.out_features));
            ops::gemm(input, weight, output, ctx);
            return;
        }

        Tensor weight = make_2d_f32_tensor_view(
            single_weight_.data,
            static_cast<int64_t>(single_weight_.out_features),
            static_cast<int64_t>(single_weight_.in_features));
        if (needs_torch_matmul_path(input, output, weight))
        {
            run_out_in_matmul(input, weight, output);
            return;
        }

        const float* input_ptr = static_cast<const float*>(tensor_data(input));
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
                    single_weight_.data,
                    in_features_,
                    out_features_total_,
                    out_col,
                    single_weight_.layout);
            }
        }
        return;
    }

    for (int32_t i = 0; i < stacked_weight_count_; ++i)
    {
        const StackedWeightDesc& desc = stacked_weights_[i];
        Tensor weight = desc.layout == WeightLayout::kInOut
            ? make_2d_f32_tensor_view(
                desc.data,
                static_cast<int64_t>(desc.in_features),
                static_cast<int64_t>(desc.out_features))
            : make_2d_f32_tensor_view(
                desc.data,
                static_cast<int64_t>(desc.out_features),
                static_cast<int64_t>(desc.in_features));

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
            continue;
        }

        const float* weight_ptr = desc.data;
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
    if (use_stacked_weights_ && (stacked_weights_ == nullptr || stacked_weight_count_ <= 0))
    {
        throw std::runtime_error("modules::Linear::forward: stacked weights are not bound.");
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
        if (desc.data == nullptr)
        {
            throw std::runtime_error("modules::Linear::bind_stacked_weights: desc.data must be non-null.");
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
