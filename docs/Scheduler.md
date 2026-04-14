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

遍历 running 中的任务，在 `max_num_scheduled_tokens` 限制下选取任务，并尽量完成（通过分析 Request 中的属性，如果当前为 Prefill，那么可以调度该 Request 所有未 Prefill 的 token；如果为 Decode，那么可以调度一个 Token），对于选取的每一个 token，使用 `kv_cache_manager` 的 `allocate_slots` 方法，根据其调度的 token 数量为其分配内存块。如果内存块不够，那么从 running 队列的末端调用 `_preempt_request` 放弃已执行的 task，直到足够分配。

如果在调度 running 队列过程中已经出现 `_preempt_request`，那么跳过 waiting 队列调度；否则按照类似策略为 waiting 队列中的 request 调度并分配必要显存。