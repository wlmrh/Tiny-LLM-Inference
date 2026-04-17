#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/scheduler.h"

int main()
{
    constexpr int32_t kNumLayers = 1;
    constexpr int32_t kBlockSizeTokens = 4;
    constexpr size_t kNumBlocks = 16;
    constexpr size_t kBlockBytes = 128;

    void* kv_pool = std::malloc(kNumBlocks * kBlockBytes);
    assert(kv_pool != nullptr);

    tiny_llm::EngineArgs args;
    args.kv_num_layers = kNumLayers;
    args.kv_block_size_tokens = kBlockSizeTokens;
    args.kv_num_blocks = kNumBlocks;
    args.kv_block_size_bytes = kBlockBytes;
    args.kv_memory_pool = kv_pool;
    args.scheduler_config.policy = tiny_llm::SchedulerPolicy::kFcfs;
    args.scheduler_config.max_prefill_tokens_per_step = 2;

    tiny_llm::Scheduler scheduler(args);

    tiny_llm::Request req;
    req.request_id = 1;
    req.prompt_token_ids = {5, 6, 7};
    req._all_token_ids = req.prompt_token_ids;
    req.status = tiny_llm::RequestStatus::WAITING;
    req.sampling_params.max_tokens = 2;
    req.sampling_params.stop_token_ids = {2};
    scheduler.add_request(req);

    assert(scheduler.get_num_unfinished_requests() == 1);
    assert(scheduler.has_unfinished_requests());

    {
        const tiny_llm::SchedulerOutput out = scheduler.schedule();
        assert(out.scheduled_new_reqs.size() == 1);
        assert(out.scheduled_cached_reqs.req_ids.empty());
        assert(out.num_scheduled_tokens.at(1) == 2);

        tiny_llm::ModelRunnerOutput model_out;
        model_out.req_ids = {1};
        model_out.req_id_to_index[1] = 0;
        model_out.sampled_token_ids = {-1};

        const std::map<int, tiny_llm::EngineCoreOutput> results =
            scheduler.update_from_output(out, model_out);
        assert(results.empty());
    }

    {
        const tiny_llm::SchedulerOutput out = scheduler.schedule();
        assert(out.scheduled_new_reqs.size() == 1);
        assert(out.num_scheduled_tokens.at(1) == 1);

        tiny_llm::ModelRunnerOutput model_out;
        model_out.req_ids = {1};
        model_out.req_id_to_index[1] = 0;
        model_out.sampled_token_ids = {-1};

        const std::map<int, tiny_llm::EngineCoreOutput> results =
            scheduler.update_from_output(out, model_out);
        assert(results.empty());
    }

    {
        const tiny_llm::SchedulerOutput out = scheduler.schedule();
        assert(out.scheduled_new_reqs.empty());
        assert(out.scheduled_cached_reqs.req_ids.size() == 1);
        assert(out.scheduled_cached_reqs.req_ids[0] == 1);
        assert(out.num_scheduled_tokens.at(1) == 1);

        tiny_llm::ModelRunnerOutput model_out;
        model_out.req_ids = {1};
        model_out.req_id_to_index[1] = 0;
        model_out.sampled_token_ids = {2};

        const std::map<int, tiny_llm::EngineCoreOutput> results =
            scheduler.update_from_output(out, model_out);
        assert(results.size() == 1);
        assert(results.at(1).new_token_id == 2);
        assert(results.at(1).generated_tokens == 1);
    }

    assert(!scheduler.has_unfinished_requests());
    assert(scheduler.get_num_unfinished_requests() == 0);

    std::free(kv_pool);

    {
        constexpr int32_t kNumLayers2 = 1;
        constexpr int32_t kBlockSizeTokens2 = 1;
        constexpr size_t kNumBlocks2 = 1;
        constexpr size_t kBlockBytes2 = 128;

        void* kv_pool2 = std::malloc(kNumBlocks2 * kBlockBytes2);
        assert(kv_pool2 != nullptr);

        tiny_llm::EngineArgs args2;
        args2.kv_num_layers = kNumLayers2;
        args2.kv_block_size_tokens = kBlockSizeTokens2;
        args2.kv_num_blocks = kNumBlocks2;
        args2.kv_block_size_bytes = kBlockBytes2;
        args2.kv_memory_pool = kv_pool2;
        args2.scheduler_config.policy = tiny_llm::SchedulerPolicy::kFcfs;
        args2.scheduler_config.max_prefill_tokens_per_step = 2;

        tiny_llm::Scheduler scheduler2(args2);

        tiny_llm::Request req1;
        req1.request_id = 1;
        req1.prompt_token_ids = {5};
        req1._all_token_ids = req1.prompt_token_ids;
        req1.status = tiny_llm::RequestStatus::WAITING;
        req1.sampling_params.max_tokens = 2;
        req1.sampling_params.stop_token_ids = {2};
        scheduler2.add_request(req1);

        tiny_llm::Request req2;
        req2.request_id = 2;
        req2.prompt_token_ids = {6};
        req2._all_token_ids = req2.prompt_token_ids;
        req2.status = tiny_llm::RequestStatus::WAITING;
        req2.sampling_params.max_tokens = 2;
        req2.sampling_params.stop_token_ids = {2};
        scheduler2.add_request(req2);

        {
            const tiny_llm::SchedulerOutput out = scheduler2.schedule();
            assert(out.scheduled_new_reqs.size() == 1);
            assert(out.scheduled_new_reqs[0].req_id == 1);

            tiny_llm::ModelRunnerOutput model_out;
            model_out.req_ids = {1};
            model_out.req_id_to_index[1] = 0;
            model_out.sampled_token_ids = {-1};
            const std::map<int, tiny_llm::EngineCoreOutput> results =
                scheduler2.update_from_output(out, model_out);
            assert(results.empty());
        }

        {
            const tiny_llm::SchedulerOutput out = scheduler2.schedule();
            // Running-phase preempt should force this step to skip waiting scheduling.
            assert(out.scheduled_new_reqs.empty());
            assert(out.scheduled_cached_reqs.req_ids.empty());
        }

        std::free(kv_pool2);
    }

    return 0;
}
