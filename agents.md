# Project Context (项目上下文)

Tiny-LLM-Inference is a small vLLM-inspired single-process inference engine with a frontend/core split: `LLMEngine` handles text/tokenizer I/O, while `EngineCore` runs scheduler + model execution over token IDs. The runtime uses FCFS continuous batching semantics across `running` and `waiting` queues, supports chunked prefill, one-token decode steps, simplified tail preemption, and paged KV-cache metadata through `SchedulerOutput`.

# Tech Stack & Tooling (技术栈与工具)

- Language standard: C++17, configured in the root `CMakeLists.txt` via `set(CMAKE_CXX_STANDARD 17)`.
- Build system: CMake 3.18+, with a single static library target `tiny_llm` and compatibility aliases `tiny_llm_core`, `tiny_llm_models`, and `tiny_llm_operators`.
- libtorch integration: `find_package(Torch QUIET)` first, then Python `torch.utils.cmake_prefix_path`, then `$HOME/libtorch*`. `${TORCH_LIBRARIES}` is linked into `tiny_llm`; `TORCH_CXX_FLAGS` is appended globally so the libtorch ABI flag is inherited by the project and third-party tokenizer code.
- CPU/CUDA mode: CPU is the default (`TINYLLM_ENABLE_CUDA=OFF`) and compiles `src/core/cpu/cpu_allocator.cpp`. CUDA mode (`TINYLLM_ENABLE_CUDA=ON`) enables CUDA language, requires `CUDAToolkit`, links `CUDA::cudart`, `CUDA::cublas`, `${TORCH_LIBRARIES}`, and compiles CUDA allocator/operator kernels.
- Tokenizer integration: the Rust-backed HuggingFace tokenizer path is provided by `tokenizers-cpp`, fetched with CMake `FetchContent` from `https://github.com/mlc-ai/tokenizers-cpp.git` and linked as `tokenizers_cpp`. CMake requires `cargo` before fetching it.
- Tokenizer wrapper path: C++ runtime wrappers live in `include/tiny_llm/runtime/tokenizer.h` and `src/runtime/tokenizer.cpp`. `HFLlamaTokenizer` loads `tokenizer.json` via `tokenizers::Tokenizer::FromBlobJSON` or `tokenizer.model` via `FromBlobSentencePiece`.

# Architecture (架构与目录)

- Runtime scheduling: `include/tiny_llm/runtime/scheduler.h` and `src/runtime/scheduler.cpp`. `Scheduler` owns request state, `waiting`/`running` queues, token budgets, chunked prefill/decode selection, and preemption. `KVCacheManager` bridges scheduling decisions to KV block allocation.
- Engine frontend/core: `include/tiny_llm/runtime/engine.h`, `src/runtime/engine.cpp`, `include/tiny_llm/runtime/engine_core.h`, and `src/runtime/engine_core.cpp`. `LLMEngine` converts user text to core requests; `EngineCore::step()` calls `Scheduler::schedule()`, `ModelExecutor::execute_model()`, then `Scheduler::update_from_output()`.
- Model execution: `include/tiny_llm/runtime/executor.h` and `src/runtime/executor.cpp`. `ModelExecutor` flattens scheduled request tokens into `input_tokens`, `position_ids`, `slot_mapping`, `context_lens`, and `block_tables`, installs paged-attention metadata, runs `Model::forward_step()`, and samples the last scheduled row per request.
- Attention: public API in `include/tiny_llm/operators/paged_attention.h`; implementation in `src/operators/paged_attention/paged_attention.cpp`; CUDA baseline kernel in `src/operators/paged_attention/paged_attention_kernels.cu`. LLaMA self-attention orchestration is in `include/tiny_llm/models/llama_decoder_layer.h` and `src/models/llama_layer.cpp`.
- KV Cache memory pool: metadata service in `include/tiny_llm/runtime/kv_cache.h` and `src/runtime/kv_cache.cpp`; physical block allocator in `include/tiny_llm/core/allocator.h` and `src/core/allocator_common.cpp`, with CPU/CUDA backend files under `src/core/cpu/` and `src/core/cuda/`.
- Model weight loading: HuggingFace config loader in `include/tiny_llm/models/hf_llama_config_loader.h` and `src/models/hf_llama_config_loader.cpp`; safetensors loader in `include/tiny_llm/models/hf_safetensors_loader.h` and `src/models/hf_safetensors_loader.cpp`; name-to-weight registry in `include/tiny_llm/models/llama_weight_map.h` and `src/models/llama_weight_map.cpp`; LLaMA model wiring in `include/tiny_llm/models/llama_model.h` and `src/models/llama_model.cpp`.
- C++/Rust wrapper layer: `src/runtime/tokenizer.cpp` conditionally includes `<tokenizers_cpp.h>`, defines a fallback C++ declaration for the external `tokenizers::Tokenizer` API, and wraps the handle inside `HFLlamaTokenizer::Impl`.
- Public headers: current public API is under `include/tiny_llm/`. Some legacy-looking headers exist under `include/core/` and `include/models/`; prefer `include/tiny_llm/...` for new code.

# Coding Conventions (代码规范)

- Tensor type: `tiny_llm::Tensor` is currently `using Tensor = torch::Tensor` in `include/tiny_llm/core/tensor.h`. Helper functions such as `tensor_dtype`, `tensor_shape`, `tensor_data`, and `make_tensor_from_blob` should be preferred over reaching through custom legacy Tensor assumptions.
- Tensor passing: read-only tensor inputs are generally passed as `const Tensor&`; mutable outputs and scratch buffers are passed as `Tensor&`; factory/load functions may return `Tensor` by value because `torch::Tensor` is an intrusive reference-counted handle. Examples: `Model::forward_step(const Tensor& input_ids, const Tensor& positions, Tensor& logits, ...)`, `ops::attention_paged(const Tensor& q, Tensor& out, ...)`, and `HFSafeTensorLoader::tensor(...) -> Tensor`.
- Tensor ownership: `torch::empty` creates owning tensors for model buffers and temporary runtime inputs. `torch::from_blob`/`make_tensor_from_blob` creates non-owning views, so the backing memory must outlive the returned `Tensor`.
- Memory management: owning polymorphic/resources are usually `std::unique_ptr` (`EngineCore`, `Scheduler`, `ModelExecutor`, owned models, HF loader, `KVCache`, tokenizer impl). Non-owning dependencies are raw pointers (`Model*`, `KVCache*`, `Tokenizer*`, `ExecutionContext*`, `StackAllocator*`) and are validated before use.
- Allocator convention: `StackAllocator` is a monotonic per-step workspace; tensors from it are non-owning and valid only until reset/begin-step. `BlockAllocator` manages persistent fixed-size KV blocks; `KVCache::end_sequence()` must release them.
- Error handling: the C++ code favors fail-fast `std::runtime_error` with contextual prefixes for invalid configuration, shape/dtype mismatch, missing files, invalid token IDs, allocator exhaustion, and tokenizer/FFI construction failures. Scheduler preemption cleanup is best-effort in a narrow path and catches exceptions to preserve forward progress.
- FFI/tokenizer handling: `HFLlamaTokenizer` stores the external tokenizer handle in a move-only pImpl (`std::unique_ptr<Impl>`). `from_model_dir()` validates model directory, tokenizer file presence, handle construction, vocab size, and special token IDs, then throws `std::runtime_error` on failure.

# Workflows (构建与测试流)

Configure and build CPU-only:

```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=OFF
cmake --build build -j
```

Configure and build CUDA:

```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=ON
cmake --build build -j
```

Manually point CMake at libtorch:

```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=OFF -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build -j
```

Use Python PyTorch as the libtorch provider:

```bash
python -c 'import torch; print(torch.utils.cmake_prefix_path)'
cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j
```

Run all CTest tests:

```bash
cd build
ctest --output-on-failure
```

Run the currently approved tiny runtime test binary directly:

```bash
./build/tests/test_tiny_lm_runtime
```

Run examples:

```bash
./build/tiny_lm_inference
./build/llama_inference
```

# Gotchas & Landmines (避坑指南)

[TODO: Human developer please fill in memory management and FFI landmines here]
