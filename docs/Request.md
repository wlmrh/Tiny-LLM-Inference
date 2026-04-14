```c++
struct Request{
	uint64_t request_id; // 该请求的序号（全局唯一）
  uint64_t priority; // 该请求的优先级
  SamplingParams sampling_params；// 该请求的采样参数
  RequestStatus status; // RequestStatus 是一个枚举类型，需要新建，包含 RUNNING 和 FINISHED 两种状态
  std::vector<int32_t> prompt_token_ids; // 输入的 token id 序列
  std::vector<int32_t> _all_token_ids; // 已经生成的 token id 序列，初始状态为用户输入的 token id 序列
}
```