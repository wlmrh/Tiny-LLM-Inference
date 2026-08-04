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
    // Scheduler-selected tokens flattened request by request for the current model step.
    Tensor input_ids; // [num_total_tokens], int32
    // Absolute sequence position corresponding to each flattened token.
    Tensor positions; // [num_total_tokens], int32
    // Physical KV-cache slot assigned to each flattened token.
    Tensor slot_mapping; // [num_total_tokens], int32
    // Batch-local request index corresponding to each flattened token.
    Tensor seq_indices; // [num_total_tokens], int32
    // Valid context length of each request after this step finishes writing K/V.
    Tensor context_lens; // [num_seqs], int32
    // Per-layer mapping from logical sequence blocks to physical KV-cache blocks.
    Tensor block_tables; // [num_layers, num_seqs, max_blocks_per_seq], int32

    // Rows whose logits should be sampled, usually the final row of each request.
    std::vector<int32_t> sample_row_offsets;

    // CPU mirror retained once per scheduler step so attention layers do not copy block tables back from CUDA.
    std::vector<int32_t> host_block_tables;
    // Describes the flattened query interval and absolute start position of every scheduled request.
    std::vector<ops::PagedAttentionQuerySegment> query_segments;
    // True when query_segments form a complete, gap-free description of input_ids.
    bool query_segments_valid = false;
};

} // namespace tiny_llm
