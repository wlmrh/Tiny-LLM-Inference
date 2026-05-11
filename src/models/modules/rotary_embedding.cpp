#include "tiny_llm/models/modules/rotary_embedding.h"

#include "tiny_llm/operators/llama_ops.h"

#include <stdexcept>

namespace tiny_llm {
namespace modules {

RotaryEmbedding::RotaryEmbedding(int32_t num_attention_heads,
                                 int32_t num_key_value_heads,
                                 int32_t head_dim,
                                 float rope_theta)
    : num_attention_heads_(num_attention_heads),
      num_key_value_heads_(num_key_value_heads),
      head_dim_(head_dim),
      rope_theta_(rope_theta)
{
    if (num_attention_heads_ <= 0 || num_key_value_heads_ <= 0 || head_dim_ <= 0 || rope_theta_ <= 0.0f)
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
        rope_theta_);
}

} // namespace modules
} // namespace tiny_llm
