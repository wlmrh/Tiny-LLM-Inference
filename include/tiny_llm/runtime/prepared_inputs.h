#pragma once

#include <cstdint>
#include <vector>

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/operators/paged_attention.h"

namespace tiny_llm {

/**
 * @brief Flattened model inputs prepared from a SchedulerOutput.
 *
 * Model code consumes this tensor package without knowing request ids, queues,
 * scheduler policy, or how the batch was assembled.
 */
struct PreparedInputs {
    Tensor input_ids;       // [num_total_tokens], int32
    Tensor positions;       // [num_total_tokens], int32
    Tensor slot_mapping;    // [num_total_tokens], int32
    Tensor seq_indices;     // [num_total_tokens], int32
    Tensor context_lens;    // [num_seqs], int32
    Tensor block_tables;    // [num_layers, num_seqs, max_blocks_per_seq], int32

    // Rows whose logits should be sampled, usually the final row of each request.
    std::vector<int32_t> sample_row_offsets;

    // Precomputed once per scheduler step to avoid per-layer CUDA metadata CPU reads.
    std::vector<ops::PagedAttentionPrefillSegment> prefill_segments;
    bool prefill_segments_valid = false;
};

} // namespace tiny_llm
