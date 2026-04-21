# Tiny-LLM-Inference

Tiny-LLM-Inference 是一个面向学习和工程验证的轻量级 LLM 推理项目，当前目标是维护一条可运行、可调试、可扩展的最小推理链路：

- 请求接入
- prefill / decode 调度
- 模型前向
- 采样与输出

## 当前状态

项目当前实现聚焦 Stage 2（Minimal Runtime Chain）：

- 运行时采用 Frontend + Core 分层：
  - `LLMEngine` 负责文本输入输出编排
  - `EngineCore` 负责调度与执行
- 调度器采用 FCFS，支持 chunked prefill 和简化抢占。
- 执行器按步构造扁平输入（不使用持久 InputBatch）：
  - `input_tokens`
  - `position_ids`
  - `slot_mapping`
  - `context_lens`
  - `block_tables`
- paged attention 目前是基线实现：
  - 前向路径保留
  - 对 metadata 一致性进行强校验
  - CUDA kernel 仍为正确性优先的 baseline
- 已支持两类 tokenizer：
  - `WhitespaceTokenizer`（动态词表）
  - `WordPieceTokenizer`（固定词表，来自 `vocab.txt`）
- 已支持 `TinyEmbeddingLM` 文本 checkpoint (`tiny_lm_checkpoint_v1`) 加载。
- 核心 Tensor 抽象已切换到 `torch::Tensor`（通过 libtorch）。

## 代码结构

- `include/tiny_llm/`
  - 公共 API（core / operators / models / runtime）
- `src/`
  - 对应实现层
- `assets/tiny_lm/`
  - `vocab.txt`
  - `tiny_lm_checkpoint.txt`
- `examples/`
  - `llama_inference.cpp`
  - `tiny_lm_inference.cpp`
- `tests/`
  - 当前启用 `test_tiny_lm_runtime.cpp`

## 构建依赖

必需：

- CMake >= 3.18
- C++17 编译器
- libtorch（C++ API）

可选：

- CUDA Toolkit（开启 `TINYLLM_ENABLE_CUDA=ON` 时）
- cuBLAS（CUDA 构建时）

说明：

- CMake 会优先 `find_package(Torch)`。
- 若失败，会尝试：
  - Python `torch.utils.cmake_prefix_path`
  - `$HOME/libtorch` 与 `$HOME/libtorch*`

## 构建

### CPU-only

```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=OFF
cmake --build build -j
```

### CUDA

```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=ON
cmake --build build -j
```

### 手动指定 Torch 路径（可选）

```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=OFF -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build -j
```

## 运行示例

### TinyEmbeddingLM + WordPiece

```bash
./build/tiny_lm_inference
```

可传入自定义参数：

```bash
./build/tiny_lm_inference assets/tiny_lm/vocab.txt assets/tiny_lm/tiny_lm_checkpoint.txt "hello"
```

### MiniLLaMA + Whitespace

```bash
./build/llama_inference
```

## 运行测试

```bash
cd build
ctest --output-on-failure
```

当前默认启用测试：

- `test_tiny_lm_runtime`

## 关键运行时契约

- `SchedulerOutput` 统一使用：
  - `scheduled_reqs`
  - `num_scheduled_tokens`
  - `total_num_scheduled_tokens`
  - `finished_req_ids`
  - `preempted_req_ids`
- `ModelExecutor` 只对每个序列本步最后一个 token 行采样。
- KV 资源分配由 Scheduler / KVCacheManager 负责，执行器不再调用 `ensure_capacity`。

## 当前边界与限制

- 单进程、单设备优先（不包含分布式执行）。
- 调度策略以最小可用 FCFS 为主。
- paged attention 计算内核为基线版本，优先契约正确性与可诊断性。

## 常见问题

### Torch not found

报错示例：`Torch not found. Set Torch_DIR/CMAKE_PREFIX_PATH to libtorch, or install Python torch.`

解决方式：

- 设置 `CMAKE_PREFIX_PATH` 或 `Torch_DIR`
- 或安装 Python `torch` 让 CMake 自动探测

### macOS 上 libtorch 依赖 libomp 路径不一致

如遇 `libtorch_cpu.dylib` 依赖 `libomp.dylib` 路径问题，可按本机实际路径修复 install name。
