#pragma once

#include <cstdint>

#include "tiny_llm/core/tensor.h"

namespace tiny_llm {
namespace modules {

class RotaryEmbedding : public torch::nn::Module {
public:
    RotaryEmbedding(int32_t num_attention_heads,
                    int32_t num_key_value_heads,
                    int32_t head_dim,
                    float rope_theta);

    void forward(const Tensor& positions, Tensor& q, Tensor& k) const;

private:
    int32_t num_attention_heads_ = 0;
    int32_t num_key_value_heads_ = 0;
    int32_t head_dim_ = 0;
    float rope_theta_ = 0.0f;
};

} // namespace modules
} // namespace tiny_llm
