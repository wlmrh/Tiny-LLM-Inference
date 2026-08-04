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
    // First row of this request in the flattened Q/K/V tensors for the current step.
    int64_t row_start = 0;
    // Batch-local sequence index used to select context lengths and block-table rows.
    int32_t seq_index = 0;
    // Absolute position of the first query; equal to the number of tokens cached before this step.
    int32_t query_start_position = 0;
    // Number of consecutive query tokens scheduled for this request in the current step.
    int32_t query_length = 0;
};

struct PagedAttentionRuntimeScratch
{
    // Step-scoped destinations used to gather paged keys and values into contiguous SDPA inputs.
    Tensor contiguous_k;
    Tensor contiguous_v;
    // One offset-causal mask per query segment, reusable by every transformer layer in this step.
    std::vector<Tensor> offset_causal_masks;
};

struct PagedAttentionRuntimeMetadata
{
    // Scheduler-prepared global KV slot for each flattened token row.
    const Tensor *slot_mapping = nullptr;
    // Maps each flattened token row to its batch-local sequence index.
    const Tensor *seq_indices = nullptr;
    // Total valid context length of each sequence after the current step's KV write.
    const Tensor *context_lens = nullptr;
    // GPU block tables with shape [num_layers, num_seqs, max_blocks_per_seq].
    const Tensor *block_tables = nullptr;
    // CPU mirror of block_tables used to assemble cached prefixes without a GPU-to-CPU sync per layer.
    const int32_t *host_block_tables = nullptr;
    int64_t host_block_table_count = 0;
    // Non-owning view of the continuous query slice belonging to each scheduled request.
    const PagedAttentionQuerySegment *query_segments = nullptr;
    int64_t query_segment_count = 0;
    // Scratch storage owned by RuntimeContext and shared by all layers in one model step.
    PagedAttentionRuntimeScratch *scratch = nullptr;
    // Number of token slots stored in one physical KV-cache block.
    int32_t block_size_tokens = 0;
    // True only when query_segments exactly and contiguously cover all flattened rows.
    bool query_segments_valid = false;
    // Enables paged-KV attention for this invocation.
    bool enabled = false;
};

struct LlamaAttentionParams
{
    // Absolute sequence position for every flattened query row.
    const Tensor *positions = nullptr;
    // Projected and RoPE-transformed query, key, and value tensors for the current layer.
    const Tensor *q = nullptr;
    const Tensor *k = nullptr;
    const Tensor *v = nullptr;
    // Receives attention results in the original flattened row order.
    Tensor *out = nullptr;
    // Provides the CUDA stream and the persistent paged KV cache.
    ExecutionContext *ctx = nullptr;
    // Scheduler-derived metadata used to map flattened rows back to sequences and KV blocks.
    const PagedAttentionRuntimeMetadata *metadata = nullptr;
    // Selects this layer's slice in the per-layer block tables and KV cache.
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
