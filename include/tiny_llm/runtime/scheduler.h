#pragma once

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

struct RequestData {
    uint64_t req_id = 0; // 请求 id
    std::vector<int32_t> new_token_ids; // 本轮要计算的新 token 集合
    int32_t num_computed_tokens = 0; // 已经计算过 kvcache 的 token 长度
    bool is_prefill = false; // Whether this scheduled chunk is still processing prompt tokens.
    std::vector<int32_t> block_ids; // 该请求的映射表，将逻辑上的 block id 映射到物理块编号
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
    std::vector<uint64_t> finished_req_ids; ///< 在上一步中完成的请求，供 ModelRunner 清除这些请求的cache
    std::vector<uint64_t> preempted_req_ids; ///< 本轮中被抢占的请求 id 序列，供 ModelRunner 释放资源
    // std::vector<int32_t> new_block_ids_to_zero; ///< 本轮新分配的，需要初始化的物理块 id（由于 persistent caching 机制，ModelRunner 中存储着每个请求执行状态的副本）
};

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
    }
};

/**
 * @brief Aggregated model execution results for one runtime step.
 */
struct ModelRunnerOutput {
    std::vector<uint64_t> req_ids; // 本轮执行的所有请求 id
    std::unordered_map<uint64_t, int32_t> req_id_to_index; // id -> index（该请求在sampled_token_ids 的第几项）
    std::vector<int32_t> sampled_token_ids; // 本轮迭代中，每个请求的产出（idx -> token_id），prefill 请求为 -1
    RuntimeProfilingStats profiling;
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

    void refresh_block_table(
        int32_t core_seq_id,
        bool kv_started,
        std::vector<int32_t>& block_table) const;
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
    KVCache* kv_cache() const { return kvcache_manager.kv_cache(); }

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
