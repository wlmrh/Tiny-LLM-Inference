#pragma once

#include <algorithm>
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

    void add(const RuntimeProfilingStats& other)
    {
        prepare_inputs_ms += other.prepare_inputs_ms;
        prefill_ms += other.prefill_ms;
        decode_ms_total += other.decode_ms_total;
        sampling_ms += other.sampling_ms;
        embedding_ms += other.embedding_ms;
        qkv_proj_ms += other.qkv_proj_ms;
        rope_ms += other.rope_ms;
        attention_ms += other.attention_ms;
        o_proj_ms += other.o_proj_ms;
        mlp_ms += other.mlp_ms;
        norm_ms += other.norm_ms;
        lm_head_ms += other.lm_head_ms;
        prefill_tokens += other.prefill_tokens;
        decode_tokens += other.decode_tokens;
        sampled_tokens += other.sampled_tokens;
        scheduled_requests += other.scheduled_requests;
        scheduled_tokens += other.scheduled_tokens;
        prefill_requests += other.prefill_requests;
        decode_requests += other.decode_requests;
        max_context_len = std::max(max_context_len, other.max_context_len);
        profiled_steps += other.profiled_steps;
    }
};

} // namespace tiny_llm
