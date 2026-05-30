#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "tiny_llm/core/tensor.h"

namespace tiny_llm {
namespace modules {

class RotaryEmbedding : public torch::nn::Module {
public:
    RotaryEmbedding(int32_t num_attention_heads,
                    int32_t num_key_value_heads,
                    int32_t head_dim,
                    float rope_theta,
                    std::string rope_scaling_type = "",
                    float rope_scaling_factor = 1.0f,
                    float rope_scaling_low_freq_factor = 1.0f,
                    float rope_scaling_high_freq_factor = 1.0f,
                    int32_t rope_scaling_original_max_position_embeddings = 0,
                    int32_t max_position_embeddings = 0);

    void forward(const Tensor& positions, Tensor& q, Tensor& k) const;

private:
    Tensor inv_freq_for_device(const c10::Device& device) const;
    std::pair<Tensor, Tensor> cos_sin_for_device(const c10::Device& device) const;

    int32_t num_attention_heads_ = 0;
    int32_t num_key_value_heads_ = 0;
    int32_t head_dim_ = 0;
    float rope_theta_ = 0.0f;
    std::string rope_scaling_type_;
    float rope_scaling_factor_ = 1.0f;
    float rope_scaling_low_freq_factor_ = 1.0f;
    float rope_scaling_high_freq_factor_ = 1.0f;
    int32_t rope_scaling_original_max_position_embeddings_ = 0;
    int32_t max_position_embeddings_ = 0;
    mutable Tensor cached_inv_freq_;
};

} // namespace modules
} // namespace tiny_llm
