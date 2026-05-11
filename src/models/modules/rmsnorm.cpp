#include "tiny_llm/models/modules/rmsnorm.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/operators/rmsnorm.h"

#include <stdexcept>

namespace tiny_llm {
namespace modules {

RMSNorm::RMSNorm(int32_t hidden_size, float eps)
    : hidden_size_(hidden_size), eps_(eps)
{
    if (hidden_size_ <= 0)
    {
        throw std::runtime_error("modules::RMSNorm: hidden_size must be positive.");
    }
    if (eps_ <= 0.0f)
    {
        throw std::runtime_error("modules::RMSNorm: eps must be > 0.");
    }
}

void RMSNorm::bind_weights(const Tensor& weight)
{
    if (!weight.defined())
    {
        throw std::runtime_error("modules::RMSNorm::bind_weights: weight tensor must be defined.");
    }
    if (tensor_dtype(weight) != DType::kFloat32)
    {
        throw std::runtime_error("modules::RMSNorm::bind_weights: weight tensor must be float32.");
    }
    if (weight.dim() != 1 || weight.size(0) != hidden_size_)
    {
        throw std::runtime_error("modules::RMSNorm::bind_weights: weight tensor shape must be [hidden_size].");
    }
    if (tensor_data(weight) == nullptr)
    {
        throw std::runtime_error("modules::RMSNorm::bind_weights: weight tensor data pointer must be non-null.");
    }

    weight_ = register_parameter("weight", weight, /*requires_grad=*/false);
}

void RMSNorm::bind_weights(float* weight)
{
    if (weight == nullptr)
    {
        throw std::runtime_error("modules::RMSNorm::bind_weights: weight pointer must be non-null.");
    }

    const auto options = torch::TensorOptions()
        .dtype(to_torch_scalar_type(DType::kFloat32))
        .device(infer_blob_device(weight));
    bind_weights(torch::from_blob(weight, {hidden_size_}, options));
}

Tensor RMSNorm::forward(const Tensor& input, ExecutionContext& ctx) const
{
    if (!input.defined())
    {
        throw std::runtime_error("modules::RMSNorm::forward: input must be defined.");
    }

    Tensor output = torch::empty_like(input);
    forward(input, output, ctx);
    return output;
}

void RMSNorm::forward(const Tensor& input, Tensor& output, ExecutionContext& ctx) const
{
    validate_forward_inputs(input, output);

    ops::rmsnorm(input, weight_, output, ctx, eps_);
}

void RMSNorm::validate_forward_inputs(const Tensor& input, const Tensor& output) const
{
    if (!weight_.defined())
    {
        throw std::runtime_error("modules::RMSNorm::forward: weights are not bound.");
    }
    if (!input.defined() || !output.defined())
    {
        throw std::runtime_error("modules::RMSNorm::forward: input and output must be defined.");
    }
    if (tensor_dtype(input) != DType::kFloat32 || tensor_dtype(output) != DType::kFloat32)
    {
        throw std::runtime_error("modules::RMSNorm::forward: only float32 tensors are supported.");
    }
    if (input.dim() < 1 || output.dim() < 1)
    {
        throw std::runtime_error("modules::RMSNorm::forward: input and output rank must be >= 1.");
    }
    if (!input.sizes().equals(output.sizes()))
    {
        throw std::runtime_error("modules::RMSNorm::forward: input and output shapes must match.");
    }
    if (input.size(input.dim() - 1) != hidden_size_)
    {
        throw std::runtime_error("modules::RMSNorm::forward: trailing dimension must equal hidden_size.");
    }
    if (tensor_data(input) == nullptr || tensor_data(output) == nullptr)
    {
        throw std::runtime_error("modules::RMSNorm::forward: input/output pointers must be non-null.");
    }
}

} // namespace modules
} // namespace tiny_llm
