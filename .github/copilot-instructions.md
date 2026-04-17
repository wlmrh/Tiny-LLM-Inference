# Tiny-LLM-Inference Copilot Instructions

本文件定义 GitHub Copilot 在本仓库中的默认行为。目标是让生成代码与当前架构、模块边界、构建方式和测试策略保持一致。

## 项目目标

Tiny-LLM-Inference 是一个面向学习与工程验证的轻量级 LLM 推理引擎，当前处于 Stage 2（Minimal Runtime Chain）。

Copilot 在本项目中的主要目标：

- 维护一个可运行的最小推理链路：请求接入、prefill、decode、采样、输出。
- 在 CPU-only 与 CUDA 两种构建模式下保持接口一致与行为可预测。
- 强化模块解耦：Core 提供基础设施，Operators 提供算子，Models 提供统一模型接口与加载实现，Runtime 负责调度。
- 以统一运行时契约对接可扩展的模型文件格式与 tokenizer，优先保证接口稳定和校验完整性。
- 优先保证可读性、契约清晰度与测试可验证性，再考虑性能优化。

当前明确边界：

- Runtime 以单进程、单设备为主。
- Runtime 调度仅面向单机单卡单线程，不引入多队列优先级系统与分布式协调。
- 已支持两类 tokenizer：Whitespace（动态词表）与 WordPiece（固定词表 `vocab.txt`）。
- 已支持 `TinyEmbeddingLM` 文本 checkpoint（`tiny_lm_checkpoint_v1`）加载路径，用于对接流程验证。
- Runtime 采用 text-only 的 InputPreprocessor / OutPreprocessor 分层，不处理多模态字段。
- Tokenizer 通过独立 `TokenizerRegistry` 管理，主要用于 InputPreprocessor 与 OutPreprocessor；EngineCore 不承担 encode/decode 与 tokenizer 契约校验。
- Runtime 拆分 Frontend 与 EngineCore：Frontend 只保留最小编排状态，EngineCore 负责调度编排与执行驱动。
- Runtime 状态下放：EngineCore 仅作为最小协调器（Scheduler + Executor），请求状态、队列与序列生命周期由 Scheduler 统一管理。
- Scheduler 采用最小可用 FCFS 机制，输出任务计划（prefill/decode），支持 chunked prefill。
- KVCacheManager 负责 KVCache 交互封装（容量估算、start/end sequence、block table 刷新），并支持基于构造参数递归创建 KVCache；EngineCore 不直接管理 KV 细节。
- EngineCore 的 step 闭环固定为：`Scheduler::schedule()` 产出任务描述，`ModelExecutor` 执行任务，`Scheduler::update_from_output(...)` 回写状态并整理资源。
- Scheduler 在 `update_from_output(...)` 阶段执行 finished request 回收，确保 KV block 及时释放（不再依赖独立 `post_step` 接口）。
- 部分 CUDA kernel 是基线实现，优先正确性和接口稳定性。

## 技术栈

- 语言与标准：C++17，CUDA C++（可选）。
- 构建系统：CMake（最低版本 3.18）。
- CUDA 开关：TINYLLM_ENABLE_CUDA（ON/OFF）。
- 关键库：CUDA Runtime，cuBLAS（仅 CUDA 模式）。
- 测试框架：CTest + 轻量断言测试（assert）。

构建目标分层：

- tiny_llm_core：张量、执行上下文、分配器、tokenizer、processors、KV 元数据与 runtime engine。
- tiny_llm_operators：rmsnorm、matmul、paged_attention 及其后端派发。
- tiny_llm_models：`MiniLLaMA` 与 `TinyEmbeddingLM`（含 checkpoint 加载）。
- llama_inference：基于 `MiniLLaMA + WhitespaceTokenizer` 的最小链路示例。
- tiny_lm_inference：基于 `TinyEmbeddingLM + WordPieceTokenizer` 的 checkpoint 驱动示例。

## 目录结构说明

请遵循以下目录职责，不要跨层混放实现：

- include/tiny_llm/
	- 对外公共 API，按 core、operators、models、runtime 分层。
	- `models/model.h` 定义统一模型契约，`runtime/tokenizer.h` 定义统一 tokenizer 契约。
- src/
	- 与 include/tiny_llm 对应的实现层。
	- `src/models/tiny_lm.cpp` 放置 checkpoint 加载与前向实现。
	- `src/runtime/tokenizer.cpp` 放置 WordPiece 与 Whitespace tokenizer 实现。
	- `src/runtime/request.cpp` 放置 `Request` / `RequestStatus` 相关实现（定义位于 `include/tiny_llm/runtime/request.h`）。
	- `src/runtime/processors.cpp` 放置 InputPreprocessor 与 OutPreprocessor 实现。
	- `src/runtime/executor.cpp` 放置 ModelExecutor（模型执行与采样）实现。
	- src/core/cpu 与 src/core/cuda 放置后端特化实现。
	- src/operators/<op> 放置算子入口与 CUDA kernel。
- assets/
	- 模型资产目录（当前示例：`assets/tiny_lm/vocab.txt`、`assets/tiny_lm/tiny_lm_checkpoint.txt`）。
- tests/
	- 单元与对比测试（CPU 参考实现对照，CUDA 数值校验、runtime 契约回归）。
- examples/
	- `llama_inference.cpp`：基础最小推理链路。
	- `tiny_lm_inference.cpp`：模型文件 + tokenizer 对接链路。
- build/
	- 构建产物目录，禁止手改。

核心模块关系（调用主链）：

1. `examples/tiny_lm_inference.cpp` 加载 `WordPieceTokenizer::from_vocab_file` 与 `TinyEmbeddingLM::from_checkpoint`。
2. `runtime/engine` 作为 Frontend（`LLMEngine`）：仅保留 `InputPreprocessor input_preprocessor_`、`OutPreprocessor output_preprocessor_`、`std::unique_ptr<EngineCore> core_` 三个成员。
3. `InputPreprocessor` 负责 prompt 翻译与校验，并完成外部请求到内部请求的 ID 标准化，生成 `EngineCoreRequest`（包含 `internal_id`、`prompt_token_ids` 与 `SamplingParams`）。
4. `runtime/engine_core` 作为协调层，仅持有 `Scheduler + ModelExecutor` 并驱动调度与执行闭环（schedule -> execute -> update）。
5. `Scheduler + KVCacheManager` 持有核心运行时状态（请求队列、`Request` 与其运行时元数据、`core_seq_id` 分配）并负责任务选择、KV block 估算与 block_table 刷新。
	Scheduler 负责构造 KVCacheManager；KVCacheManager 可递归构造 KVCache；KVCache 可递归构造 BlockAllocator。
	调度结果为可执行任务描述包，EngineCore 负责执行任务并将执行结果回传给 Scheduler 做状态回写。
6. `ModelExecutor`（Executor）负责实际 `Model::forward_step` 执行与采样，隔离执行细节。
7. `OutPreprocessor` 维护 `RequestState`，执行增量 decode、停止条件判定（EOS/stop_token/length）并返回用户输出。
8. `operators` 执行 gemm / paged_attention / rmsnorm 等算子并做 CPU/CUDA 派发。
9. `runtime/kv_cache` 维护 page table 元数据并向 `BlockAllocator` 申请/释放物理块。

公共头文件约定：

- 优先包含 include/tiny_llm/... 下的头文件。
- include/core/... 与 include/models/... 主要是兼容转发包装，新增代码不应依赖这些旧路径。

## 核心开发约定

### 1) 模型与 tokenizer 接口契约

- 新模型必须实现 `Model` 接口：`num_layers()`、`vocab_size()`、`forward_step(...)`。
- 若模型对 special token 有约束，需覆盖 `expected_bos_id/eos_id/unk_id`。
- 新 tokenizer 必须实现 `Tokenizer` 接口：`encode/decode`、`vocab_size`、`bos/eos/unk`、`is_fixed_vocab`、`is_valid_token_id`。
- 固定词表 tokenizer（如 WordPiece）必须保证 `vocab_size` 稳定且可复现；动态词表 tokenizer 必须显式返回 `is_fixed_vocab=false`。

### 2) 模型文件格式与加载契约

- 模型加载逻辑放在 `src/models/`，不得混入 runtime 或 operators。
- 文件解析必须执行严格校验：magic/version、维度、元素数量、special token 范围、I/O 失败场景。
- 解析错误统一抛出 `std::runtime_error`，消息包含函数前缀（例如 `TinyEmbeddingLM::from_checkpoint`）。
- 资产路径默认放在 `assets/<model_name>/`，示例与测试应支持相同路径约定。

### 3) 接口与内存契约

- Tensor 是非 owning 视图，不负责内存释放。
- StackAllocator 分配的临时内存仅在当前 step 有效；begin_step 或 reset 后旧指针视为失效。
- KVCache 只管理映射元数据，不负责真实计算；请求结束必须调用 end_sequence 归还物理块。
- 新增 API 时必须同步声明、定义和调用方，避免接口漂移。

### 4) Runtime 调度与契约校验

- `LLMEngine` 的公开接口只保留：`add_request`、`has_unfinished_requests`、`step`，成员只保留 `InputPreprocessor input_preprocessor_`、`OutPreprocessor output_preprocessor_`、`std::unique_ptr<EngineCore> core_`。
- `EngineArgs` 用于聚合 `Model/ExecutionContext/Tokenizer` 句柄或其构造参数、调度默认参数以及 KV 递归构造参数（层数、block token 大小、block 数、block 字节数、内存池指针），减少构造链路中的重复传参；不得借此改变 `LLMEngine`/`EngineCore` 的成员归属或职责边界。
- `EngineCore` 的公开接口只保留：`add_request`、`step`，且 `step` 返回 `std::tuple<std::unordered_map<int, EngineCoreOutputs>, bool>`，其中布尔值表示 `SchedulerOutput.total_num_scheduled_tokens > 0`。
- `EngineCore` 私有成员仅保留 `std::unique_ptr<Scheduler> scheduler_` 与 `std::unique_ptr<ModelExecutor> executor_`，其余运行时状态不得回流到 EngineCore。
- `EngineCore::step` 必须按三段式执行：先调 `Scheduler::schedule()` 获取 `scheduler_output`，再由 `ModelExecutor` 执行获得 `model_output`，最后调 `Scheduler::update_from_output(scheduler_output, model_output)` 回写并生成 `EngineCoreOutputs`。
- prompt token、sampled token 必须同时满足 tokenizer 与 model 的 ID 范围约束。
- decode 终止条件至少覆盖 EOS 与最大生成步数，且终止时必须释放 KV sequence 资源。
- `InputPreprocessor` 公开接口只保留：`process_inputs`（负责分配 internal_id 并绑定 external_id）。
- `OutPreprocessor` 公开接口只保留：`add_request`、`process_outputs`、`has_unfinished_requests`。
- `TokenizerRegistry` 负责 tokenizer 生命周期与依赖注入，Processor 不直接持有 tokenizer 所有权；EngineCore 不承担 tokenizer 语义校验。
- `Request` 遵循 `docs/Request.md`：`request_id`、`priority`、`sampling_params`、`status`、`prompt_token_ids`、`_all_token_ids`。
- 运行时状态归属：`Request` 与其运行时元数据（`core_seq_id/arrival_order/num_computed/kv_started/block_table`）、running/waiting 队列统一归 Scheduler 管理。

### 4.2) 最小调度契约（单机单卡）

- Scheduler 的核心输出是 `SchedulerResult.tasks`，每个任务声明：请求 ID、是否 prefill、本步 token 数。
- Runtime `SchedulerOutput` 采用分层输出：`scheduled_new_reqs`、`scheduled_cached_reqs`、`num_scheduled_tokens`、`finished_req_ids`。
- FCFS 可在显存压力下执行简化抢占；未抢占请求保持在 running 队列等待后续 step。
- chunked prefill 是默认路径：
	`max_prefill_tokens_per_step` 控制每步 prefill 规模，避免超长 prompt 长时间阻塞。
- 不在 Runtime 内引入复杂 QoS、跨设备调度、异步流水线；这类能力属于后续阶段。

### 4.1) Text-only 请求/输出协议（对齐 vLLM 思路）

- `EngineCoreRequest`：必须包含 `internal_id`、`prompt_token_ids`、`SamplingParams`，文本模式下不引入多模态字段；`internal_id` 由 `InputPreprocessor::process_inputs` 统一分配。
- `RequestState`：由 OutPreprocessor 持有，至少包含 `generated_token_ids`、`decoded_prefix_len`、`is_finished`。
- `EngineCoreOutputs`：EngineCore 每步输出原始 token 增量；OutPreprocessor 在检查 stop 条件后可直接标记 sequence 完成，由 Scheduler 在后续调度/回收流程中统一清理。
- 若新增 stop 规则（如字符串停止词），优先扩展 OutPreprocessor，避免污染模型执行路径。

### 5) CPU/CUDA 双后端规则

- 所有后端分支统一使用 TINYLLM_ENABLE_CUDA 宏做条件编译。
- C++ 算子入口负责参数校验与派发，CUDA kernel 放在 .cu 中。
- CPU fallback 需要与 CUDA 版本保持形状、dtype 与输出语义一致。
- CPU-only 构建必须可通过编译；不得在公共头中引入未受保护的 CUDA-only 依赖。

### 6) 校验与错误处理

- 对外或跨模块入口函数必须进行基础校验：空指针、dtype、shape、维度上界。
- 错误统一使用 std::runtime_error，消息包含模块前缀与明确原因。
- 不要静默吞掉关键错误；资源不足、维度非法等应立即失败并提供可诊断信息。

### 7) 代码组织与风格

- 命名沿用现有风格：类型 PascalCase，函数与局部变量 snake_case，命名空间 tiny_llm 与 tiny_llm::ops。
- 新增实现按模块就近落位，避免在无关目录添加“临时实现”。
- 只做与需求相关的最小改动，不重排无关代码、不大面积格式化。
- 注释聚焦设计意图与边界条件，避免解释显而易见的语句。

### 8) 测试与验收

- 修改 core 或 runtime 时，至少补充或更新对应 tests 用例。
- 新增算子逻辑时：
	- 在 CPU 路径提供参考验证或与参考实现对比。
	- 若涉及 CUDA kernel，补充 CUDA 数值一致性测试。
- 新增模型加载或 tokenizer 逻辑时，至少补充一条端到端 runtime 测试（参考 `test_tiny_lm_runtime`）。
- 交付前至少覆盖以下验证：
	- CPU-only 构建与测试。
	- CUDA 构建与测试（若改动涉及 CUDA 或算子派发）。

### 9) Copilot 生成代码时的优先级

当需求存在多种实现路径时，按以下优先级决策：

1. 先保证契约正确性（shape/dtype/lifecycle）。
2. 再保证 model/tokenizer 与 runtime 调度链路的一致性。
3. 再保证与现有模块边界一致（core/operators/models/runtime/assets/tests/examples）。
4. 再补齐测试与可诊断错误信息。
5. 最后考虑性能优化或工程美化。

