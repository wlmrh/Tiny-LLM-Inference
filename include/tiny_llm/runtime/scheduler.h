#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tiny_llm/runtime/request.h"
#include "tiny_llm/runtime/processors.h"
#include "tiny_llm/runtime/parallel_config.h"

namespace tiny_llm {

struct EngineArgs;
class KVCache;
struct ModelRunnerOutput;

/**
 * @brief Config-driven scheduler settings.
 */
struct SchedulerConfig {
    size_t max_running_requests = 0; // 0 means no explicit limit
    bool enable_preemption = true;
    int32_t max_prefill_tokens_per_step = 256;
};

struct RequestData {
    uint64_t req_id = 0;
    std::vector<int32_t> new_token_ids;
    int32_t num_computed_tokens = 0;
    int32_t prompt_token_count = 0;
    bool is_prefill = false;
    std::vector<std::vector<int32_t>> block_tables; // [layer][logical_block] -> physical block id
    SamplingParams sampling_params;
    std::vector<int32_t> all_token_ids;
};

/**
 * @brief Scheduler output package for one runtime step.
 */
struct SchedulerOutput {
    std::vector<RequestData> scheduled_reqs;
    std::unordered_map<uint64_t, int32_t> num_scheduled_tokens;
    int32_t total_num_scheduled_tokens = 0;
    std::vector<uint64_t> preempted_req_ids;
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

    std::map<int, EngineCoreOutput> update_from_output(
        const SchedulerOutput& scheduler_output,
        const ModelRunnerOutput& model_runner_output);

    void add_request(Request request);

    int get_num_unfinished_requests() const;

    bool has_unfinished_requests() const;
    KVCache* kv_cache() const { return kvcache_manager.kv_cache(); }

private:
    void preempt_request(uint64_t request_id);
    RequestData make_request_data(
        const Request& request,
        bool is_prefill,
        int32_t scheduled_tokens) const;
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
