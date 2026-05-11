#pragma once

#include <cstdint>

#include "tiny_llm/core/tensor.h"

namespace tiny_llm {

class ExecutionContext;

namespace modules {

class RMSNorm : public torch::nn::Module {
public:
    RMSNorm(int32_t hidden_size, float eps);

    void bind_weights(const Tensor& weight);
    void bind_weights(float* weight);
    Tensor forward(const Tensor& input, ExecutionContext& ctx) const;
    void forward(const Tensor& input, Tensor& output, ExecutionContext& ctx) const;

    int32_t hidden_size() const { return hidden_size_; }
    float eps() const { return eps_; }

private:
    void validate_forward_inputs(const Tensor& input, const Tensor& output) const;

    Tensor weight_;
    int32_t hidden_size_ = 0;
    float eps_ = 0.0f;
};

} // namespace modules
} // namespace tiny_llm
