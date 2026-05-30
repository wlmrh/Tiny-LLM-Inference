#pragma once

#include <cstdint>

#include "tiny_llm/core/tensor.h"

namespace tiny_llm {
namespace ops {

void embedding_lookup(const Tensor& ids,
                      const Tensor& embedding,
                      Tensor& out,
                      int32_t vocab_size,
                      int32_t hidden_size,
                      bool embedding_is_vocab_hidden);

void split_qkv(const Tensor& qkv,
               Tensor& q,
               Tensor& k,
               Tensor& v,
               int32_t hidden_size,
               int32_t kv_hidden_size);

void apply_rope(const Tensor& positions,
                Tensor& q,
                Tensor& k,
                int32_t num_attention_heads,
                int32_t num_key_value_heads,
                int32_t head_dim,
                float rope_theta,
                const char* rope_scaling_type = "",
                float rope_scaling_factor = 1.0f,
                float rope_scaling_low_freq_factor = 1.0f,
                float rope_scaling_high_freq_factor = 1.0f,
                int32_t rope_scaling_original_max_position_embeddings = 0);

void apply_rope(const Tensor& positions,
                Tensor& q,
                Tensor& k,
                int32_t num_attention_heads,
                int32_t num_key_value_heads,
                int32_t head_dim,
                const Tensor& inv_freq);

void apply_rope(const Tensor& positions,
                Tensor& q,
                Tensor& k,
                int32_t num_attention_heads,
                int32_t num_key_value_heads,
                int32_t head_dim,
                const Tensor& cos_cache,
                const Tensor& sin_cache);

void silu_multiply(const Tensor& gate, const Tensor& up, Tensor& out);

void copy_tensor(const Tensor& src, Tensor& dst);

void add_tensors(const Tensor& lhs, const Tensor& rhs, Tensor& out);

} // namespace ops
} // namespace tiny_llm
