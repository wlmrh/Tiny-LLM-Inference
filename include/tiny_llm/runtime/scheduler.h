#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tiny_llm/runtime/request.h"
#include "tiny_llm/runtime/processors.h"

namespace tiny_llm {

struct EngineArgs;
class KVCache;

enum class SchedulerPolicy {
    kFcfs = 0,
};

/**
 * @brief Config-driven scheduler settings.
 */
struct SchedulerConfig {
    SchedulerPolicy policy = SchedulerPolicy::kFcfs;
    size_t max_running_requests = 0; // 0 means no explicit limit
    bool enable_preemption = true;
    int32_t max_prefill_tokens_per_step = 256;
};

struct NewRequestData {
    uint64_t req_id = 0;
    int32_t core_seq_id = -1;
    std::vector<int32_t> prompt_token_ids;
    std::vector<int32_t> block_ids;
    int32_t num_computed_tokens = 0;
    SamplingParams sampling_params;
};

struct CachedRequestData {
    std::vector<uint64_t> req_ids;
    std::vector<int32_t> core_seq_ids;
    std::vector<int32_t> input_token_ids;
    std::vector<std::optional<std::vector<int32_t>>> new_block_ids;
    std::vector<int32_t> num_computed_tokens;
    std::unordered_set<uint64_t> resumed_req_ids;
};

/**
 * @brief Scheduler output package for one runtime step.
 */
struct SchedulerOutput {
    std::vector<NewRequestData> scheduled_new_reqs;
    CachedRequestData scheduled_cached_reqs;
    std::unordered_map<uint64_t, int32_t> num_scheduled_tokens;
    std::vector<uint64_t> finished_req_ids;
    std::vector<uint64_t> preempted_req_ids;
    int32_t total_num_scheduled_tokens = 0;
};

/**
 * @brief Per-task execution result returned by ModelExecutor stage.
 */
struct ModelTaskOutput {
    uint64_t internal_id = 0;
    bool is_prefill = false;
    int32_t processed_tokens = 0;
    int32_t sampled_token_id = -1;
    bool has_error = false;
    std::string error_message;
};

/**
 * @brief Aggregated model execution results for one runtime step.
 */
struct ModelOutput {
    std::vector<ModelTaskOutput> tasks;
};

using ModelRunnerOutput = ModelOutput;
using EngineCoreOutput = EngineCoreOutputs;

/**
 * @brief KV block estimation and block table refresh helper for scheduler/runtime.
 */
class KVCacheManager {
public:
    KVCacheManager() = default;
    explicit KVCacheManager(KVCache* kv);
    KVCacheManager(int32_t kv_num_layers,
                   int32_t kv_block_size_tokens,
                   size_t kv_num_blocks,
                   size_t kv_block_size_bytes,
                   void* kv_memory_pool);
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
                    void* kv_memory_pool);

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

    void refresh_block_table(
        int32_t core_seq_id,
        bool kv_started,
        std::vector<int32_t>& block_table) const;

    bool allocate_slots(
        int32_t core_seq_id,
        bool kv_started,
        int32_t num_computed_tokens,
        int32_t num_new_tokens) const;

private:
    std::unique_ptr<KVCache> owned_kv_;
    KVCache* kv_ = nullptr;
};

/**
 * @brief Scheduler mechanism with strategy and configuration hooks.
 */
class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config = SchedulerConfig{});
    explicit Scheduler(const EngineArgs& args);
    Scheduler(KVCache* kv, SchedulerConfig config = SchedulerConfig{});

    SchedulerOutput schedule();

    // 根据 scheduler 的调度结果及其执行结果 model_runner_output 修改 scheduler 中 Request 的状态
    std::map<int, EngineCoreOutput> update_from_output(
        SchedulerOutput scheduler_output,
        ModelRunnerOutput model_runner_output);

    // 将 Request 中的属性补齐并添加到 waiting 队列的最后
    void add_request(Request request);

    // 获取未完成的 Request 的总数
    int get_num_unfinished_requests();

    // 返回是否有未完成的 Request
    bool has_unfinished_requests();

private:
    void _preempt_request(Request request);

    KVCacheManager kvcache_manager;
    std::map<int64_t, Request> requests;
    SchedulerPolicy policy = SchedulerPolicy::kFcfs;
    std::deque<uint64_t> waiting;
    std::deque<uint64_t> running;
    int64_t max_num_scheduled_tokens = 256;
};

} // namespace tiny_llm
