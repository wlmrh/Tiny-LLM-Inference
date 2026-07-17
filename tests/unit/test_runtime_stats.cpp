#include "tiny_llm/runtime/runtime_stats.h"

#include <gtest/gtest.h>

TEST(RuntimeProfilingStatsTest, AddAggregatesCountersAndKeepsMaxContextLen)
{
    tiny_llm::RuntimeProfilingStats total;
    total.prepare_inputs_ms = 1.0;
    total.model_ms_total = 1.5;
    total.prefill_ms = 2.0;
    total.decode_ms_total = 3.0;
    total.mixed_model_ms = 3.5;
    total.sampling_ms = 4.0;
    total.embedding_ms = 5.0;
    total.qkv_proj_ms = 6.0;
    total.rope_ms = 7.0;
    total.attention_ms = 8.0;
    total.o_proj_ms = 9.0;
    total.mlp_ms = 10.0;
    total.norm_ms = 11.0;
    total.lm_head_ms = 12.0;
    total.prefill_tokens = 13;
    total.decode_tokens = 14;
    total.sampled_tokens = 15;
    total.scheduled_requests = 16;
    total.scheduled_tokens = 17;
    total.prefill_requests = 18;
    total.decode_requests = 19;
    total.max_context_len = 20;
    total.profiled_steps = 21;

    tiny_llm::RuntimeProfilingStats step;
    step.prepare_inputs_ms = 0.5;
    step.model_ms_total = 1.0;
    step.prefill_ms = 1.5;
    step.decode_ms_total = 2.5;
    step.mixed_model_ms = 2.0;
    step.sampling_ms = 3.5;
    step.embedding_ms = 4.5;
    step.qkv_proj_ms = 5.5;
    step.rope_ms = 6.5;
    step.attention_ms = 7.5;
    step.o_proj_ms = 8.5;
    step.mlp_ms = 9.5;
    step.norm_ms = 10.5;
    step.lm_head_ms = 11.5;
    step.prefill_tokens = 1;
    step.decode_tokens = 2;
    step.sampled_tokens = 3;
    step.scheduled_requests = 4;
    step.scheduled_tokens = 5;
    step.prefill_requests = 6;
    step.decode_requests = 7;
    step.max_context_len = 42;
    step.profiled_steps = 8;

    total.add(step);

    EXPECT_DOUBLE_EQ(total.prepare_inputs_ms, 1.5);
    EXPECT_DOUBLE_EQ(total.model_ms_total, 2.5);
    EXPECT_DOUBLE_EQ(total.prefill_ms, 3.5);
    EXPECT_DOUBLE_EQ(total.decode_ms_total, 5.5);
    EXPECT_DOUBLE_EQ(total.mixed_model_ms, 5.5);
    EXPECT_DOUBLE_EQ(total.sampling_ms, 7.5);
    EXPECT_DOUBLE_EQ(total.embedding_ms, 9.5);
    EXPECT_DOUBLE_EQ(total.qkv_proj_ms, 11.5);
    EXPECT_DOUBLE_EQ(total.rope_ms, 13.5);
    EXPECT_DOUBLE_EQ(total.attention_ms, 15.5);
    EXPECT_DOUBLE_EQ(total.o_proj_ms, 17.5);
    EXPECT_DOUBLE_EQ(total.mlp_ms, 19.5);
    EXPECT_DOUBLE_EQ(total.norm_ms, 21.5);
    EXPECT_DOUBLE_EQ(total.lm_head_ms, 23.5);
    EXPECT_EQ(total.prefill_tokens, 14);
    EXPECT_EQ(total.decode_tokens, 16);
    EXPECT_EQ(total.sampled_tokens, 18);
    EXPECT_EQ(total.scheduled_requests, 20);
    EXPECT_EQ(total.scheduled_tokens, 22);
    EXPECT_EQ(total.prefill_requests, 24);
    EXPECT_EQ(total.decode_requests, 26);
    EXPECT_EQ(total.max_context_len, 42);
    EXPECT_EQ(total.profiled_steps, 29);
}
