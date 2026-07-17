#include "tiny_llm/runtime/runtime_stats.h"

#include <algorithm>

namespace tiny_llm
{

void RuntimeProfilingStats::add(const RuntimeProfilingStats &other)
{
    prepare_inputs_ms += other.prepare_inputs_ms;
    model_ms_total += other.model_ms_total;
    prefill_ms += other.prefill_ms;
    decode_ms_total += other.decode_ms_total;
    mixed_model_ms += other.mixed_model_ms;
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

} // namespace tiny_llm
