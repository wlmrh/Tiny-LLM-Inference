#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tiny_llm/runtime/model_runner_output.h"
#include "tiny_llm/runtime/processors.h"
#include "tiny_llm/runtime/request.h"
#include "tiny_llm/runtime/scheduler_config.h"

namespace tiny_llm {

struct EngineArgs;
class KVCacheManager;
class KVCache;

struct RequestData {
    uint64_t req_id = 0; // 请求 id
    std::vector<int32_t> new_token_ids; // 本轮要计算的新 token 集合
    int32_t num_computed_tokens = 0; // 已经计算过 kvcache 的 token 长度
    int32_t prompt_token_count = 0; // Request prompt 长度
    std::vector<std::vector<int32_t>> block_tables; // [layer][logical_block] -> physical block id
    SamplingParams sampling_params; // 该 Requsest 的采样参数
    std::vector<int32_t> all_token_ids; // 当前所有 token 的 id 序列
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
 * @brief Scheduler mechanism for FCFS request admission and token budgeting.
 */
class Scheduler {
public:
    explicit Scheduler(const EngineArgs& args);
    Scheduler(KVCache* kv, SchedulerConfig config = SchedulerConfig{});
    ~Scheduler();

    SchedulerOutput schedule();

    // 根据 scheduler 的调度结果及其执行结果 model_runner_output 修改 scheduler 中 Request 的状态
    std::map<int, EngineCoreOutput> update_from_output(
        const SchedulerOutput& scheduler_output,
        const ModelRunnerOutput& model_runner_output);

    // 将 Request 中的属性补齐并添加到 waiting 队列的最后
    void add_request(Request request);

    // 返回是否有未完成的 Request
    bool has_unfinished_requests() const;
    KVCache* kv_cache() const;

private:
    explicit Scheduler(SchedulerConfig config);

    void preempt_request(uint64_t request_id);

    std::unique_ptr<KVCacheManager> kvcache_manager;
    std::map<int64_t, Request> requests;
    std::deque<uint64_t> waiting;
    std::deque<uint64_t> running;
    int64_t max_num_scheduled_tokens = 256;
    size_t max_running_requests = 0;
    bool enable_preemption = true;
};

} // namespace tiny_llm
