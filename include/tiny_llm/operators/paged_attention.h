#pragma once

#include <cstdint>
#include <vector>

#include "tiny_llm/core/tensor.h"

namespace tiny_llm
{
class ExecutionContext;

namespace ops
{

struct PagedAttentionQuerySegment
{
    int64_t row_start = 0;
    int32_t seq_index = 0;
    int32_t query_start_position = 0;
    int32_t query_length = 0;
};

struct PagedAttentionRuntimeScratch
{
    Tensor contiguous_k;
    Tensor contiguous_v;
    std::vector<Tensor> offset_causal_masks;
};

struct PagedAttentionRuntimeMetadata
{
    const Tensor *slot_mapping = nullptr;
    const Tensor *seq_indices = nullptr;
    const Tensor *context_lens = nullptr;
    const Tensor *block_tables = nullptr;
    const int32_t *host_block_tables = nullptr;
    int64_t host_block_table_count = 0;
    const PagedAttentionQuerySegment *query_segments = nullptr;
    int64_t query_segment_count = 0;
    PagedAttentionRuntimeScratch *scratch = nullptr;
    int32_t block_size_tokens = 0;
    bool query_segments_valid = false;
    bool enabled = false;
};

struct LlamaAttentionParams
{
    const Tensor *positions = nullptr;
    const Tensor *q = nullptr;
    const Tensor *k = nullptr;
    const Tensor *v = nullptr;
    Tensor *out = nullptr;
    ExecutionContext *ctx = nullptr;
    const PagedAttentionRuntimeMetadata *metadata = nullptr;
    int32_t layer_id = -1;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
};

void llama_attention_forward(const LlamaAttentionParams &params);
void llama_attention(const Tensor &positions, const Tensor &q, const Tensor &k, const Tensor &v, Tensor &out,
                     ExecutionContext &ctx, int32_t layer_id, int32_t num_attention_heads, int32_t num_key_value_heads,
                     int32_t head_dim);
} // namespace ops

} // namespace tiny_llm
