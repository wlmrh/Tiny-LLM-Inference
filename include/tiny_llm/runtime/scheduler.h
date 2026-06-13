#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tiny_llm/runtime/request.h"
#include "tiny_llm/runtime/processors.h"
#include "tiny_llm/runtime/parallel_config.h"
#include "tiny_llm/runtime/model_runner_output.h"
#include "tiny_llm/runtime/scheduler_config.h"

namespace tiny_llm {

struct EngineArgs;
class KVCache;

struct RequestData {
    uint64_t req_id = 0; // 请求 id
    std::vector<int32_t> new_token_ids; // 本轮要计算的新 token 集合
    int32_t num_computed_tokens = 0; // 已经计算过 kvcache 的 token 长度
    int32_t prompt_token_count = 0;
    std::vector<std::vector<int32_t>> block_tables; // [layer][logical_block] -> physical block id
    SamplingParams sampling_params;
    std::vector<int32_t> all_token_ids;
};

/**
 * @brief Scheduler output package for one runtime step.
 */
struct SchedulerOutput {
    std::vector<RequestData> scheduled_reqs;
    std::unordered_map<uint64_t, int32_t> num_scheduled_tokens; ///< 每个 request 调度的 token 数量
    int32_t total_num_scheduled_tokens = 0; ///< 本轮调度中，所有请求要处理的 Token 总和
};

/**
 * @brief KV block estimation and block table refresh helper for scheduler/runtime.
 */
class KVCacheManager {
public:
    KVCacheManager() = default;
    explicit KVCacheManager(KVCache* kv);
    ~KVCacheManager();

    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;
    KVCacheManager(KVCacheManager&&) noexcept = default;
    KVCacheManager& operator=(KVCacheManager&&) noexcept = default;

    void bind(KVCache* kv);
    void init_owned(int32_t kv_num_layers,
                    int32_t kv_block_size_tokens,
                    size_t kv_num_blocks,
                    size_t kv_block_size_bytes,
                    void* kv_memory_pool,
                    ParallelConfig parallel_config);

    size_t free_block_count() const;
    int32_t num_layers() const;

    void start_sequence(int32_t core_seq_id) const;
    void end_sequence(int32_t core_seq_id) const;

    size_t estimate_append_new_blocks(
        int32_t core_seq_id,
        bool kv_started,
        int32_t num_computed) const;

    size_t estimate_prefill_new_blocks(
        int32_t core_seq_id,
        bool kv_started,
        int32_t prompt_tokens,
        int32_t num_computed,
        int32_t prefill_tokens) const;

    void refresh_block_tables(
        int32_t core_seq_id,
        bool kv_started,
        std::vector<std::vector<int32_t>>& block_tables) const;

    bool allocate_slots(
        int32_t core_seq_id,
        bool kv_started,
        int32_t num_computed_tokens,
        int32_t num_new_tokens) const;

    KVCache* kv_cache() const { return kv_; }

private:
    std::unique_ptr<KVCache> owned_kv_;
    KVCache* kv_ = nullptr;
};

/**
 * @brief Scheduler mechanism for FCFS request admission and token budgeting.
 */
class Scheduler {
public:
    explicit Scheduler(const EngineArgs& args);
    Scheduler(KVCache* kv, SchedulerConfig config = SchedulerConfig{});

    SchedulerOutput schedule();

    // 根据 scheduler 的调度结果及其执行结果 model_runner_output 修改 scheduler 中 Request 的状态
    std::map<int, EngineCoreOutput> update_from_output(
        SchedulerOutput scheduler_output,
        ModelRunnerOutput model_runner_output);

    // 将 Request 中的属性补齐并添加到 waiting 队列的最后
    void add_request(Request request);

    // 返回是否有未完成的 Request
    bool has_unfinished_requests();
    KVCache* kv_cache() const { return kvcache_manager.kv_cache(); }

private:
    explicit Scheduler(SchedulerConfig config);

    void _preempt_request(Request request);
    size_t running_request_count() const;
    bool can_admit_waiting_request() const;

    KVCacheManager kvcache_manager;
    std::map<int64_t, Request> requests;
    std::deque<uint64_t> waiting;
    std::deque<uint64_t> running;
    int64_t max_num_scheduled_tokens = 256;
    size_t max_running_requests = 0;
    bool enable_preemption = true;
};

} // namespace tiny_llm
