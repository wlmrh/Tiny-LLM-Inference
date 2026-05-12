#include "tiny_llm/models/modules/rotary_embedding.h"

#include "tiny_llm/operators/llama_ops.h"

#include <stdexcept>
#include <utility>

namespace tiny_llm {
namespace modules {

RotaryEmbedding::RotaryEmbedding(int32_t num_attention_heads,
                                 int32_t num_key_value_heads,
                                 int32_t head_dim,
                                 float rope_theta,
                                 std::string rope_scaling_type,
                                 float rope_scaling_factor,
                                 float rope_scaling_low_freq_factor,
                                 float rope_scaling_high_freq_factor,
                                 int32_t rope_scaling_original_max_position_embeddings)
    : num_attention_heads_(num_attention_heads),
      num_key_value_heads_(num_key_value_heads),
      head_dim_(head_dim),
      rope_theta_(rope_theta),
      rope_scaling_type_(std::move(rope_scaling_type)),
      rope_scaling_factor_(rope_scaling_factor),
      rope_scaling_low_freq_factor_(rope_scaling_low_freq_factor),
      rope_scaling_high_freq_factor_(rope_scaling_high_freq_factor),
      rope_scaling_original_max_position_embeddings_(rope_scaling_original_max_position_embeddings)
{
    if (num_attention_heads_ <= 0 || num_key_value_heads_ <= 0 || head_dim_ <= 0 || rope_theta_ <= 0.0f
        || rope_scaling_factor_ <= 0.0f)
    {
        throw std::runtime_error("modules::RotaryEmbedding: invalid configuration.");
    }
}

void RotaryEmbedding::forward(const Tensor& positions, Tensor& q, Tensor& k) const
{
    ops::apply_rope(
        positions,
        q,
        k,
        num_attention_heads_,
        num_key_value_heads_,
        head_dim_,
        rope_theta_,
        rope_scaling_type_.c_str(),
        rope_scaling_factor_,
        rope_scaling_low_freq_factor_,
        rope_scaling_high_freq_factor_,
        rope_scaling_original_max_position_embeddings_);
}

} // namespace modules
} // namespace tiny_llm
