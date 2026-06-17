#pragma once

#include <cstdint>

namespace tiny_llm {

/**
 * @brief Fine-grained runtime timing for one engine step or one generation.
 */
struct RuntimeProfilingStats {
    double prepare_inputs_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms_total = 0.0;
    double sampling_ms = 0.0;
    double embedding_ms = 0.0;
    double qkv_proj_ms = 0.0;
    double rope_ms = 0.0;
    double attention_ms = 0.0;
    double o_proj_ms = 0.0;
    double mlp_ms = 0.0;
    double norm_ms = 0.0;
    double lm_head_ms = 0.0;
    int64_t prefill_tokens = 0;
    int64_t decode_tokens = 0;
    int64_t sampled_tokens = 0;
    int64_t scheduled_requests = 0;
    int64_t scheduled_tokens = 0;
    int64_t prefill_requests = 0;
    int64_t decode_requests = 0;
    int64_t max_context_len = 0;
    int64_t profiled_steps = 0;

    void add(const RuntimeProfilingStats& other);
};

} // namespace tiny_llm
