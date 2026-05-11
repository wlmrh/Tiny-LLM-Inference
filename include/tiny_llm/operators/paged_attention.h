#pragma once

#include "tiny_llm/core/tensor.h"

namespace tiny_llm {
class ExecutionContext;

namespace ops {

struct PagedAttentionRuntimeMetadata {
    const Tensor* slot_mapping = nullptr;
    const Tensor* seq_indices = nullptr;
    const Tensor* context_lens = nullptr;
    const Tensor* block_tables = nullptr;
    int32_t block_size_tokens = 0;
    bool enabled = false;
};

struct LlamaAttentionParams {
    const Tensor* positions = nullptr;
    const Tensor* q = nullptr;
    const Tensor* k = nullptr;
    const Tensor* v = nullptr;
    Tensor* out = nullptr;
    ExecutionContext* ctx = nullptr;
    const PagedAttentionRuntimeMetadata* metadata = nullptr;
    int32_t layer_id = -1;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
};

void set_paged_attention_runtime_metadata(const Tensor& slot_mapping,
                                          const Tensor& seq_indices,
                                          const Tensor& context_lens,
                                          const Tensor& block_tables,
                                          int32_t block_size_tokens);

void clear_paged_attention_runtime_metadata();
const PagedAttentionRuntimeMetadata& current_paged_attention_runtime_metadata();

class PagedAttentionRuntimeMetadataGuard {
public:
    explicit PagedAttentionRuntimeMetadataGuard(const PagedAttentionRuntimeMetadata& metadata);
    ~PagedAttentionRuntimeMetadataGuard();

    PagedAttentionRuntimeMetadataGuard(const PagedAttentionRuntimeMetadataGuard&) = delete;
    PagedAttentionRuntimeMetadataGuard& operator=(const PagedAttentionRuntimeMetadataGuard&) = delete;

private:
    PagedAttentionRuntimeMetadata previous_{};
};

void attention_paged(const Tensor& q, Tensor& out, ExecutionContext& ctx);
void llama_attention_forward(const LlamaAttentionParams& params);
void llama_attention(const Tensor& positions,
                     const Tensor& q,
                     const Tensor& k,
                     const Tensor& v,
                     Tensor& out,
                     ExecutionContext& ctx,
                     int32_t layer_id,
                     int32_t num_attention_heads,
                     int32_t num_key_value_heads,
                     int32_t head_dim);
}

} // namespace tiny_llm
