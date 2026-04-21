# Tiny-LLM-Inference Copilot Instructions

本文件定义 Copilot 在本仓库中的默认行为。目标是让生成代码严格对齐当前代码结构、运行时契约、构建方式与测试入口。

## 1. 项目定位

Tiny-LLM-Inference 是一个面向学习与工程验证的轻量级 LLM 推理引擎，当前处于 Stage 2（Minimal Runtime Chain）。

优先级顺序：

1. 契约正确性（shape / dtype / 生命周期）
2. 运行时链路一致性（schedule -> execute -> update）
3. 模块边界清晰（core / operators / models / runtime）
4. 可测试与可诊断
5. 性能优化

## 2. 当前真实架构（以代码为准）

### 2.1 Tensor 与后端

- `include/tiny_llm/core/tensor.h` 中，`Tensor` 已是 `torch::Tensor` 别名。
- 统一通过 helper 访问：
  - `tensor_dtype`
  - `tensor_shape`
  - `tensor_data`
  - `tensor_numel`
  - `make_tensor_from_blob`
- 不要再按旧版自定义 Tensor 类（`data()/shape()/dtype()` 成员）写新代码。

### 2.2 Runtime 分层

- Frontend: `LLMEngine`
  - `add_request`
  - `has_unfinished_requests`
  - `step`
- Core: `EngineCore`
  - `add_request`
  - `step`
- 处理器：
  - `InputPreprocessor`: 文本到 `EngineCoreRequest`
  - `OutPreprocessor`: token 到用户可读输出

调用闭环固定为：

1. `Scheduler::schedule()`
2. `ModelExecutor::execute_model(...)`
3. `Scheduler::update_from_output(...)`

### 2.3 Scheduler 契约

`SchedulerOutput` 使用统一结构：

- `scheduled_reqs` (`RequestData`)
- `num_scheduled_tokens`
- `total_num_scheduled_tokens`
- `finished_req_ids`
- `preempted_req_ids`

`RequestData` 关键字段：

- `req_id`
- `new_token_ids`
- `num_computed_tokens`
- `block_ids`

说明：

- KV 资源分配由 `Scheduler` / `KVCacheManager` 完成。
- Executor 不负责 KV 容量分配。

### 2.4 ModelExecutor 契约

`ModelExecutor` 每步基于 `SchedulerOutput` 构造以下输入：

- `input_tokens` `[num_total_tokens]`
- `position_ids` `[num_total_tokens]`
- `slot_mapping` `[num_total_tokens]`
- `context_lens` `[num_seqs]`
- `block_tables` `[num_seqs, max_blocks_per_seq]`

并配合：

- `core_seq_ids`（辅助校验）
- `req_end_offsets`（每个序列本步末 token 行偏移）

采样规则：

- 只对每个序列末行（`end_offset - 1`）采样。
- 非末行采样位应保持为 `-1`。

### 2.5 Paged Attention 现状

`ops::attention_paged` 当前是正确性优先基线实现：

- 支持运行时 metadata 注入：
  - `set_paged_attention_runtime_metadata(...)`
  - `clear_paged_attention_runtime_metadata()`
- 在前向时验证 metadata 一致性（shape / dtype / block 覆盖 / slot 合法性）。
- CUDA kernel 仍为 baseline 路径（不是高性能最终实现）。

## 3. 目录职责

- `include/tiny_llm/`: 公共接口（唯一推荐 include 路径）
- `src/core/`: 执行上下文、分配器与后端特化
- `src/operators/`: 算子入口与 CUDA kernel
- `src/models/`: 模型实现与 checkpoint 加载
- `src/runtime/`: 请求生命周期、调度、执行、前后处理
- `assets/tiny_lm/`: 示例词表与 checkpoint
- `examples/`: 端到端示例
- `tests/`: 当前启用 runtime 端到端回归

## 4. 构建与测试约束

### 4.1 构建

- CMake >= 3.18，C++17。
- 依赖 libtorch。
- `TINYLLM_ENABLE_CUDA=ON` 时依赖 CUDA Toolkit + cuBLAS。

### 4.2 Torch 发现策略

CMake 会按以下顺序尝试：

1. `find_package(Torch)`
2. Python `torch.utils.cmake_prefix_path`
3. `$HOME/libtorch` 与 `$HOME/libtorch*`

### 4.3 测试

- 当前 tests/CMakeLists 仅启用 `test_tiny_lm_runtime`。
- 修改 runtime 链路（scheduler/executor/processors/engine）后，至少要保证该测试通过。

## 5. 开发硬性规则

### 5.1 接口一致性

任何 API 变更必须同步三处：

- 声明（header）
- 定义（source）
- 调用方

### 5.2 错误处理

- 统一抛 `std::runtime_error`。
- 错误消息必须包含函数前缀和可定位原因。
- 禁止静默吞掉关键失败。

### 5.3 变更范围

- 仅做需求相关最小改动。
- 不重排无关代码，不做无关格式化。
- 不要引入与当前阶段无关的分布式或复杂调度系统。

### 5.4 运行时边界

- text-only 请求协议，不引入多模态字段。
- 单机单进程单设备优先。
- 若要扩展 stop 规则，优先改 `OutPreprocessor`，不要污染模型执行路径。

## 6. 对 Copilot 的执行要求

当用户提出修改请求时：

1. 先核对当前代码实现，再修改文档或代码。
2. 文档必须反映“当前真实实现”，不保留过时设计描述。
3. 对 runtime 改动优先检查：
   - `EngineCore` 闭环完整性
   - `SchedulerOutput` 契约一致性
   - `ModelExecutor` 输入构造与采样行为
   - paged attention metadata 注入/清理是否成对
4. 交付前至少给出构建与测试结果。
