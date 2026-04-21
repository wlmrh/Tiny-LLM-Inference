#pragma once

#include "tiny_llm/core/tensor.h"

namespace tiny_llm {
class ExecutionContext;

namespace ops {
void set_paged_attention_runtime_metadata(const Tensor& slot_mapping,
										  const Tensor& context_lens,
										  const Tensor& block_tables,
										  int32_t block_size_tokens);

void clear_paged_attention_runtime_metadata();

void attention_paged(const Tensor& q, Tensor& out, ExecutionContext& ctx);
}

} // namespace tiny_llm
