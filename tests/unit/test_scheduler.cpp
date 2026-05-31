#include "tiny_llm/runtime/scheduler.h"
#include "tiny_llm/runtime/kv_cache.h"
#include "tiny_llm/core/allocator.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
struct SchedulerFixture {
    static constexpr int32_t kNumLayers = 1;
    static constexpr int32_t kBlockSizeTokens = 2;
    static constexpr size_t kBlockBytes = 64;

    std::vector<unsigned char> pool;
    tiny_llm::BlockAllocator blocks;
    tiny_llm::KVCache kv;
    tiny_llm::Scheduler scheduler;

    explicit SchedulerFixture(int32_t max_prefill_tokens = 8,
                              size_t num_blocks = 16,
                              size_t max_running_requests = 0,
                              bool enable_preemption = true)
        : pool(num_blocks * kBlockBytes),
          blocks(num_blocks, kBlockBytes, pool.data(), tiny_llm::ParallelConfig::cpu()),
          kv(make_kv_config(), &blocks),
          scheduler(&kv, make_scheduler_config(max_prefill_tokens, max_running_requests, enable_preemption))
    {
    }

    static tiny_llm::KVCache::Config make_kv_config()
    {
        tiny_llm::KVCache::Config cfg;
        cfg.num_layers = kNumLayers;
        cfg.block_size_tokens = kBlockSizeTokens;
        return cfg;
    }

    static tiny_llm::SchedulerConfig make_scheduler_config(int32_t max_prefill_tokens,
                                                           size_t max_running_requests,
                                                           bool enable_preemption)
    {
        tiny_llm::SchedulerConfig cfg;
        cfg.max_prefill_tokens_per_step = max_prefill_tokens;
        cfg.max_running_requests = max_running_requests;
        cfg.enable_preemption = enable_preemption;
        return cfg;
    }

    void add_request(uint64_t id,
                     std::vector<int32_t> prompt,
                     int32_t max_tokens = 4,
                     std::vector<int32_t> stop_token_ids = {})
    {
        tiny_llm::Request req;
        req.request_id = id;
        req.prompt_token_ids = std::move(prompt);
        req.sampling_params.max_tokens = max_tokens;
        req.sampling_params.stop_token_ids = std::move(stop_token_ids);
        scheduler.add_request(std::move(req));
    }
};

tiny_llm::ModelRunnerOutput sampled(std::initializer_list<std::pair<uint64_t, int32_t>> values)
{
    tiny_llm::ModelRunnerOutput output;
    for (const auto& item : values)
    {
        const int32_t index = static_cast<int32_t>(output.req_ids.size());
        output.req_ids.push_back(item.first);
        output.sampled_token_ids.push_back(item.second);
        output.req_id_to_index[item.first] = index;
    }
    return output;
}
}

TEST(SchedulerTest, SchedulesWaitingRequestsInFcfsOrder)
{
    SchedulerFixture fixture;
    fixture.add_request(1, {10});
    fixture.add_request(2, {20});

    tiny_llm::SchedulerOutput output = fixture.scheduler.schedule();
    ASSERT_EQ(output.scheduled_reqs.size(), 2u);
    EXPECT_EQ(output.scheduled_reqs[0].req_id, 1u);
    EXPECT_EQ(output.scheduled_reqs[1].req_id, 2u);
    EXPECT_EQ(output.total_num_scheduled_tokens, 2);
}

TEST(SchedulerTest, RespectsMaxRunningRequestAdmissionLimit)
{
    SchedulerFixture fixture(8, 16, 1);
    fixture.add_request(1, {10}, 2);
    fixture.add_request(2, {20}, 1);

    tiny_llm::SchedulerOutput first = fixture.scheduler.schedule();
    ASSERT_EQ(first.scheduled_reqs.size(), 1u);
    EXPECT_EQ(first.scheduled_reqs[0].req_id, 1u);
    auto first_outputs = fixture.scheduler.update_from_output(first, sampled({{1, 11}}));
    ASSERT_EQ(first_outputs.size(), 1u);

    tiny_llm::SchedulerOutput second = fixture.scheduler.schedule();
    ASSERT_EQ(second.scheduled_reqs.size(), 1u);
    EXPECT_EQ(second.scheduled_reqs[0].req_id, 1u);
    auto second_outputs = fixture.scheduler.update_from_output(second, sampled({{1, 12}}));
    ASSERT_EQ(second_outputs.size(), 1u);

    tiny_llm::SchedulerOutput third = fixture.scheduler.schedule();
    ASSERT_EQ(third.scheduled_reqs.size(), 1u);
    EXPECT_EQ(third.scheduled_reqs[0].req_id, 2u);
}

TEST(SchedulerTest, ChunksPrefillAndEmitsSampleWhenPromptCompletes)
{
    SchedulerFixture fixture(2);
    fixture.add_request(1, {10, 11, 12}, 3);

    tiny_llm::SchedulerOutput first = fixture.scheduler.schedule();
    ASSERT_EQ(first.scheduled_reqs.size(), 1u);
    EXPECT_EQ(first.scheduled_reqs[0].new_token_ids, std::vector<int32_t>({10, 11}));
    auto first_outputs = fixture.scheduler.update_from_output(first, sampled({{1, 40}}));
    EXPECT_TRUE(first_outputs.empty());
    EXPECT_TRUE(fixture.scheduler.has_unfinished_requests());

    tiny_llm::SchedulerOutput second = fixture.scheduler.schedule();
    ASSERT_EQ(second.scheduled_reqs.size(), 1u);
    EXPECT_EQ(second.scheduled_reqs[0].num_computed_tokens, 2);
    EXPECT_EQ(second.scheduled_reqs[0].new_token_ids, std::vector<int32_t>({12}));
    auto second_outputs = fixture.scheduler.update_from_output(second, sampled({{1, 41}}));
    ASSERT_EQ(second_outputs.size(), 1u);
    EXPECT_EQ(second_outputs.at(1).new_token_id, 41);
    EXPECT_EQ(second_outputs.at(1).generated_tokens, 1);
}

TEST(SchedulerTest, DecodeStepConsumesLastGeneratedTokenAndStopsByLength)
{
    SchedulerFixture fixture(8);
    fixture.add_request(1, {5}, 2);

    tiny_llm::SchedulerOutput prefill = fixture.scheduler.schedule();
    auto prefill_outputs = fixture.scheduler.update_from_output(prefill, sampled({{1, 6}}));
    ASSERT_EQ(prefill_outputs.at(1).new_token_id, 6);

    tiny_llm::SchedulerOutput decode = fixture.scheduler.schedule();
    ASSERT_EQ(decode.scheduled_reqs.size(), 1u);
    EXPECT_EQ(decode.scheduled_reqs[0].num_computed_tokens, 1);
    EXPECT_EQ(decode.scheduled_reqs[0].new_token_ids, std::vector<int32_t>({6}));
    auto decode_outputs = fixture.scheduler.update_from_output(decode, sampled({{1, 7}}));
    ASSERT_EQ(decode_outputs.size(), 1u);
    EXPECT_EQ(decode_outputs.at(1).new_token_id, 7);
    EXPECT_EQ(decode_outputs.at(1).generated_tokens, 2);
    EXPECT_FALSE(fixture.scheduler.has_unfinished_requests());
}

TEST(SchedulerTest, StopsByStopTokenAndCleansFinishedRequest)
{
    SchedulerFixture fixture;
    fixture.add_request(1, {5}, 8, {9});

    tiny_llm::SchedulerOutput prefill = fixture.scheduler.schedule();
    auto outputs = fixture.scheduler.update_from_output(prefill, sampled({{1, 9}}));
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs.at(1).new_token_id, 9);
    EXPECT_FALSE(fixture.scheduler.has_unfinished_requests());
    EXPECT_EQ(fixture.scheduler.get_num_unfinished_requests(), 0);

    tiny_llm::SchedulerOutput empty = fixture.scheduler.schedule();
    EXPECT_TRUE(empty.scheduled_reqs.empty());
    EXPECT_EQ(empty.total_num_scheduled_tokens, 0);
}


TEST(SchedulerTest, PreemptedRequestRecomputesPromptAndGeneratedContext)
{
    SchedulerFixture fixture(8, 2);
    fixture.add_request(1, {10, 11}, 2);
    fixture.add_request(2, {20, 21}, 3);

    tiny_llm::SchedulerOutput prefill = fixture.scheduler.schedule();
    ASSERT_EQ(prefill.scheduled_reqs.size(), 2u);
    auto prefill_outputs = fixture.scheduler.update_from_output(prefill, sampled({{1, 100}, {2, 200}}));
    ASSERT_EQ(prefill_outputs.size(), 2u);

    tiny_llm::SchedulerOutput decode_first = fixture.scheduler.schedule();
    ASSERT_EQ(decode_first.scheduled_reqs.size(), 1u);
    ASSERT_EQ(decode_first.preempted_req_ids.size(), 1u);
    EXPECT_EQ(decode_first.preempted_req_ids[0], 2u);
    EXPECT_EQ(decode_first.scheduled_reqs[0].req_id, 1u);
    auto decode_outputs = fixture.scheduler.update_from_output(decode_first, sampled({{1, 101}}));
    ASSERT_EQ(decode_outputs.size(), 1u);
    EXPECT_TRUE(fixture.scheduler.has_unfinished_requests());

    tiny_llm::SchedulerOutput recompute = fixture.scheduler.schedule();
    ASSERT_EQ(recompute.scheduled_reqs.size(), 1u);
    EXPECT_EQ(recompute.scheduled_reqs[0].req_id, 2u);
    EXPECT_TRUE(recompute.scheduled_reqs[0].is_prefill);
    EXPECT_EQ(recompute.scheduled_reqs[0].num_computed_tokens, 0);
    EXPECT_EQ(recompute.scheduled_reqs[0].new_token_ids, std::vector<int32_t>({20, 21, 200}));
}

TEST(SchedulerTest, DoesNotPreemptWhenPreemptionIsDisabled)
{
    SchedulerFixture fixture(8, 2, 0, false);
    fixture.add_request(1, {10, 11}, 3);
    fixture.add_request(2, {20, 21}, 3);

    tiny_llm::SchedulerOutput prefill = fixture.scheduler.schedule();
    ASSERT_EQ(prefill.scheduled_reqs.size(), 2u);
    auto prefill_outputs = fixture.scheduler.update_from_output(prefill, sampled({{1, 100}, {2, 200}}));
    ASSERT_EQ(prefill_outputs.size(), 2u);

    tiny_llm::SchedulerOutput decode = fixture.scheduler.schedule();
    EXPECT_TRUE(decode.scheduled_reqs.empty());
    EXPECT_TRUE(decode.preempted_req_ids.empty());
    EXPECT_EQ(decode.total_num_scheduled_tokens, 0);
    EXPECT_TRUE(fixture.scheduler.has_unfinished_requests());
}
