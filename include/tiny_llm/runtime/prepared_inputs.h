#pragma once

#include <cstdint>
#include <vector>

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/operators/paged_attention.h"

namespace tiny_llm
{

/**
 * @brief Flattened model inputs prepared from a SchedulerOutput.
 *
 * Model code consumes this tensor package without knowing request ids, queues,
 * scheduler policy, or how the batch was assembled.
 */
struct PreparedInputs
{
    Tensor input_ids;    // [num_total_tokens], int32, current token id
    Tensor positions;    // [num_total_tokens], int32, offset of current token in the request
    Tensor slot_mapping; // [num_total_tokens], int32, global KVCache index in global KV slot
    Tensor seq_indices;  // [num_total_tokens], int32
    Tensor context_lens; // [num_seqs], int32, total caluculated token after this step
    Tensor block_tables; // [num_layers, num_seqs, max_blocks_per_seq], int32, logical page number projection

    // Rows whose logits should be sampled, usually the final row of each request.
    std::vector<int32_t> sample_row_offsets;

    std::vector<int32_t> host_block_tables;
    std::vector<ops::PagedAttentionQuerySegment> query_segments;
    bool query_segments_valid = false;
};

} // namespace tiny_llm
