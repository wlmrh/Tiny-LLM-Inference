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
    uint64_t req_id = 0; // 该请求的 id
    int32_t core_seq_id = -1;
    std::vector<int32_t> prompt_token_ids; // 该请求的 token id 序列
    std::vector<int32_t> block_ids; // 该请求拥有的所有 kv block 序号
    int32_t num_computed_tokens = 0; // 该请求已经计算的 token 数量
    SamplingParams sampling_params; // 采样参数
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
    std::vector<NewRequestData> scheduled_new_reqs; // 首次调度的请求对象
    CachedRequestData scheduled_cached_reqs; // 之前已经调度过的请求，它们的数据已经 cached
    std::unordered_map<uint64_t, int32_t> num_scheduled_tokens; // 每个 request 调度的 token 数量
    std::vector<uint64_t> finished_req_ids; // 在上一步中完成的请求，供 ModelRunner 清除这些请求的cache
    int32_t total_num_scheduled_tokens = 0; // 本轮调度中，所有请求要处理的 Token 总和
    // std::vector<int32_t> new_block_ids_to_zero; // 本轮新分配的，需要初始化的物理块 id（由于 persistent caching 机制，ModelRunner 中存储着每个请求执行状态的副本）
};

/**
 * @brief Aggregated model execution results for one runtime step.
 */
struct ModelRunnerOutput {
    std::vector<uint64_t> req_ids; // 本轮执行的所有请求 id
    std::unordered_map<uint64_t, int32_t> req_id_to_index; // id -> index（该请求在sampled_token_ids 的第几项）
    std::vector<int32_t> sampled_token_ids; // 本轮迭代中，每个请求的产出（idx -> token_id），prefill 请求为 -1
    // std::vector<ModelTaskOutput> tasks;
};

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
