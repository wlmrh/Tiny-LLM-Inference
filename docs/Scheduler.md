## class Scheduler

Scheduler 类维护所有的推理任务，并负责调度每一轮 step 中执行哪些任务，具体属性、接口如下：

```c++
class Scheduler {
public:
    SchedulerOutput schedule();

    std::map<int, EngineCoreOutput> update_from_output(
        SchedulerOutput scheduler_output, 
        ModelRunnerOutput model_runner_output
    );

    /**
     * @brief
     * 将外部 InputProcessor 处理好的请求对象放入 waiting 队列。
     * @param request 包含 Token IDs 和采样参数的请求实体。
     */
    void add_request(Request request);

    /**
     * @brief
     * 获取当前系统中所有未完成请求的总数（Waiting + Running）。
     */
    int get_num_unfinished_requests();

    /**
     * @brief
     * 判断是否还有未完成的任务
     */
    bool has_unfinished_requests();

private:
    KVCacheManager kvcache_manager;     ///< 负责管理 PagedAttention 的物理块池的类
		void _preempt_request(Request request); ///< 释放 request 对应的资源，并将其添加到 waiting 的最前端
    std::map<int64_t, Request> requests; ///< 请求索引表，id -> Request
    
    SchedulingPolicy policy;            ///< 调度策略对象：实现 FCFS、优先级或其他自定义调度算法。
    
    std::deque<Request> waiting;       ///< 等待队列：存放新进入系统没有被调度的请求。
    std::deque<Request> running;       ///< 运行队列：存放已经被调度执行，但是还没有完成的。
  	int64_t max_num_scheduled_tokens;  ///< 单次调度中，新增的 token 数量上限
};
```

`schedule()` 方法调度逻辑如下：

首先进行 token 选取。在该步中，遍历 `running` 队列中的任务，在 `max_num_scheduled_tokens` 限制下选取任务，并尽量完成（对于每一个 Request，如果当前为 Prefill 阶段，那么尝试调度该 Request 所有未完成 Prefill 的 token；如果为 Decode，那么尝试调度一个 Token）。

然后进行内存分配。对于选取的每一个 token，尝试使用 `kv_cache_manager` 的 `allocate_slots` 方法，为新调度的任务分配内存块。如果内存块不够，那么从 running 队列的末端调用 `_preempt_request` 放弃已执行的 task，直到足够分配。

如果在调度 running 队列过程中已经出现 `_preempt_request`，那么跳过 waiting 队列调度；否则按照类似策略为 waiting 队列中的 request 调度并分配必要显存。

---

作为 `schedule()` 方法的返回值，`ReqeustOuput`需要包含：`scheduled_seqs`被调度的请求列表；`num_scheduled_tokens` 每个请求被调度的 token 数量，用于在 Attention 计算时，准确地从一维输出切片出每一个请求的部分；`total_num_scheduled_tokens` 调度的总 token 数量，用于在 ModelRunner 中统一分配内存而不用频繁 append；`finished_req_ids` 通知底层的KVCache Manager 释放他们占用的物理块，以便后续请求调用。

```c++
struct RequestData {
    // 1. 请求 ID
    // 对应 vLLM: NewRequestData.req_id / CachedRequestData.req_ids
    // 注：vLLM 使用 std::string，为了单机 C++ 性能，你使用 uint64_t 是极其明智的优化。
    uint64_t req_id; 

    // 2. 本轮要计算的新 Token 集合
    // 对应 vLLM: CachedRequestData.new_token_ids / NewRequestData.prefill_token_ids
    // 含义：融合了 prefill 和 decode。
    // 如果是 prefill，这里就是 prompt_token_ids；如果是 decode，这里就是上一轮生成的那个 token。
    std::vector<int32_t> new_token_ids; 

    // 3. 历史上下文长度（非常关键的对齐！）
    // 对应 vLLM: NewRequestData.num_computed_tokens / CachedRequestData.num_computed_tokens
    // 含义：之前被叫作 context_len，vLLM 中统一称为 num_computed_tokens（已经计算过 KV Cache 的 token 数量）。
    // 这名字更精确，ModelRunner 就是用它来作为 Position ID 的起点。
    int32_t num_computed_tokens; 

    // 4. 物理块映射表
    // 对应 vLLM: NewRequestData.block_ids
    // 含义：之前叫 block_table。因为你不需要像 CachedRequestData 那样只传 diff (new_block_ids)，
    // 你可以直接传当前请求完整的 block_ids 列表，供 ModelRunner 算 slot_mapping。
    std::vector<int32_t> block_ids; 

    // ---------------------------------------------------------
    // 以下为（针对你当前进度的）可选对齐属性：

    // (可选) 已经生成的 Output Token 数量
    // 对应 vLLM: CachedRequestData.num_output_tokens
    // 含义：方便后续判断请求是否达到了 max_new_tokens 上限，对于极简调度器目前不是强依赖。
    // int32_t num_output_tokens = 0; 
};

/**
 * @brief Scheduler output package for one runtime step.
 */
struct SchedulerOutput {
    std::vector<RequestData> scheduled_reqs; // 当前轮次中调度的所有请求对象
  
    std::unordered_map<uint64_t, int32_t> num_scheduled_tokens; // 每个 request 调度的 token 数量
    int32_t total_num_scheduled_tokens = 0; // 本轮调度中，所有请求要处理的 Token 总和
  
    std::vector<uint64_t> finished_req_ids; // 在上一步中完成的请求，供 ModelRunner 清除这些请求的cache
    std::vector<uint64_t> preempted_req_ids; // 通知系统这些请求被抢占，需要释放块或退回等待队列
};
```

