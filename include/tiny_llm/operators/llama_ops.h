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
                float rope_theta);

void silu_multiply(const Tensor& gate, const Tensor& up, Tensor& out);

void copy_tensor(const Tensor& src, Tensor& dst);

void add_tensors(const Tensor& lhs, const Tensor& rhs, Tensor& out);

} // namespace ops
} // namespace tiny_llm
