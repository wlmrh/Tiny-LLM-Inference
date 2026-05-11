# Project Context (项目上下文)

Tiny-LLM-Inference is a small vLLM-inspired single-process inference engine with a frontend/core split: `LLMEngine` handles text/tokenizer I/O, while `EngineCore` runs scheduler + model execution over token IDs. The runtime uses FCFS continuous batching semantics across `running` and `waiting` queues, supports chunked prefill, one-token decode steps, simplified tail preemption, and paged KV-cache metadata through `SchedulerOutput`.

# Tech Stack & Tooling (技术栈与工具)

- Language standard: C++17, configured in the root `CMakeLists.txt` via `set(CMAKE_CXX_STANDARD 17)`.
- Build system: CMake 3.18+, with a single static library target `tiny_llm` and compatibility aliases `tiny_llm_core`, `tiny_llm_models`, and `tiny_llm_operators`.
- libtorch integration: `find_package(Torch QUIET)` first, then Python `torch.utils.cmake_prefix_path`, then `$HOME/libtorch*`. `${TORCH_LIBRARIES}` is linked into `tiny_llm`; `TORCH_CXX_FLAGS` is appended globally so the libtorch ABI flag is inherited by the project and third-party tokenizer code.
- CPU/CUDA mode: CPU is the default (`TINYLLM_ENABLE_CUDA=OFF`) and compiles `src/core/cpu/cpu_allocator.cpp`. CUDA mode (`TINYLLM_ENABLE_CUDA=ON`) enables CUDA language, requires `CUDAToolkit`, links `CUDA::cudart`, `CUDA::cublas`, `${TORCH_LIBRARIES}`, and compiles CUDA allocator/operator kernels. CUDA builds must still keep CPU as the default runtime unless `EngineArgs::parallel_config` or a tool option explicitly selects CUDA.
- Tokenizer integration: the Rust-backed HuggingFace tokenizer path is provided by `tokenizers-cpp`, fetched with CMake `FetchContent` from `https://github.com/mlc-ai/tokenizers-cpp.git` and linked as `tokenizers_cpp`. CMake requires `cargo` before fetching it.
- FetchContent update policy: after the initial tokenizers-cpp population, CMake should not update/fetch it on every configure. Keep `FETCHCONTENT_UPDATES_DISCONNECTED` enabled so repeated local/remote configures do not depend on GitHub availability.
- Tokenizer wrapper path: C++ runtime wrappers live in `include/tiny_llm/runtime/tokenizer.h` and `src/runtime/tokenizer.cpp`. `HFLlamaTokenizer` loads `tokenizer.json` via `tokenizers::Tokenizer::FromBlobJSON` or `tokenizer.model` via `FromBlobSentencePiece`.

# Architecture (架构与目录)

- Test/tool layout: automated CTest entry points live under `tests/unit/` and `tests/integration/`; standalone C++ debugging runtimes live under `tools/`; Python comparison/debug helpers live under `scripts/`. Keep new diagnostic executables out of `tests/` unless they are the actual test entry point.
- Runtime scheduling: `include/tiny_llm/runtime/scheduler.h` and `src/runtime/scheduler.cpp`. `Scheduler` owns request state, `waiting`/`running` queues, token budgets, chunked prefill/decode selection, preemption, and the runtime `KVCache` used by generation. `KVCacheManager` bridges scheduling decisions to KV block allocation and can return all per-layer block tables for scheduled requests.
- Engine frontend/core: `include/tiny_llm/runtime/engine.h`, `src/runtime/engine.cpp`, `include/tiny_llm/runtime/engine_core.h`, and `src/runtime/engine_core.cpp`. `LLMEngine` converts user text to core requests; `EngineCore::step()` calls `Scheduler::schedule()`, `ModelRunner::run()`, then `Scheduler::update_from_output()`. `EngineCore` passes the scheduler-owned `KVCache*` into `ModelRunner`; do not create a second runner-local KV cache for the same engine.
- Model execution: `include/tiny_llm/runtime/model_runner.h` and `src/runtime/model_runner.cpp`. `ModelRunner` flattens scheduled request tokens into `PreparedInputs` (`input_ids`, `positions`, `slot_mapping`, `seq_indices`, `context_lens`, rank-3 `block_tables`, and `sample_row_offsets`), creates an explicit `RuntimeContext`, runs `Model::forward(const PreparedInputs&, RuntimeContext&)`, then greedily samples the request-final rows. The legacy `ModelExecutor`/`forward_step()` path has been removed.
- Model layer split: public model APIs live under `include/tiny_llm/models/`. `Model` is the causal-LM runtime interface; `LlamaForCausalLM` owns `LlamaModel` plus the LM-head projection; `LlamaModel` owns embeddings, decoder layers, and final norm. Reusable building blocks live under `include/tiny_llm/models/modules/` and currently include `Embedding`, `Linear`, `RMSNorm`, and `RotaryEmbedding`. Keep request lifecycle, batch layout, and sampling out of model modules.
- Attention: public API in `include/tiny_llm/operators/paged_attention.h`; implementation in `src/operators/paged_attention/paged_attention.cpp`; CUDA baseline kernel in `src/operators/paged_attention/paged_attention_kernels.cu`. LLaMA self-attention orchestration is in `include/tiny_llm/models/llama_decoder_layer.h` and `src/models/llama_layer.cpp`. LLaMA attention reads paged metadata from `RuntimeContext::attention_metadata()`, uses a scoped `PagedAttentionRuntimeMetadataGuard` while calling the existing operator, and with `ExecutionContext::kv()` writes/reads persistent KV blocks for runtime prefill/decode. Without KV metadata it falls back to in-batch causal attention for direct model tools such as `llama_tensor_dump`.
- CUDA paged attention: the current CUDA `llama_attention` path is a correctness bridge. It accepts CUDA tensors, copies small control metadata to CPU for validation/indexing, wraps KV blocks with `torch::from_blob(... device=kv_cache.device())`, writes current K/V into paged KV cache, gathers context K/V by `block_tables`/`seq_indices`/positions, and computes attention via libtorch `matmul + softmax + matmul`. Do not treat this as the final optimized paged-attention kernel.
- KV Cache memory pool: metadata service in `include/tiny_llm/runtime/kv_cache.h` and `src/runtime/kv_cache.cpp`; physical block allocator in `include/tiny_llm/core/allocator.h` and `src/core/allocator_common.cpp`, with CPU/CUDA backend files under `src/core/cpu/` and `src/core/cuda/`. CPU LLaMA KV data is float32 and stored per physical block as `[K block][V block]`, where each side is `block_size_tokens * kv_hidden_size` floats and `kv_hidden_size = num_key_value_heads * head_dim`.
- Model weight loading: HuggingFace config loader in `include/tiny_llm/models/hf_llama_config_loader.h` and `src/models/hf_llama_config_loader.cpp`; safetensors loader in `include/tiny_llm/models/hf_safetensors_loader.h` and `src/models/hf_safetensors_loader.cpp`; name-to-weight registry in `include/tiny_llm/models/llama_weight_map.h` and `src/models/llama_weight_map.cpp`; LLaMA model wiring in `include/tiny_llm/models/llama_model.h` and `src/models/llama_model.cpp`.
- C++/Rust wrapper layer: `src/runtime/tokenizer.cpp` conditionally includes `<tokenizers_cpp.h>`, defines a fallback C++ declaration for the external `tokenizers::Tokenizer` API, and wraps the handle inside `HFLlamaTokenizer::Impl`.
- Public headers: current public API is under `include/tiny_llm/`. Some legacy-looking headers exist under `include/core/` and `include/models/`; prefer `include/tiny_llm/...` for new code.

# Coding Conventions (代码规范)

- Tensor type: `tiny_llm::Tensor` is currently `using Tensor = torch::Tensor` in `include/tiny_llm/core/tensor.h`. Helper functions such as `tensor_dtype`, `tensor_shape`, `tensor_data`, and `make_tensor_from_blob` should be preferred over reaching through custom legacy Tensor assumptions.
- Tensor passing: read-only tensor inputs are generally passed as `const Tensor&`; mutable outputs and scratch buffers are passed as `Tensor&`; factory/load functions may return `Tensor` by value because `torch::Tensor` is an intrusive reference-counted handle. Examples: `Model::forward(const PreparedInputs& inputs, RuntimeContext& ctx) -> Tensor`, `ops::attention_paged(const Tensor& q, Tensor& out, ...)`, and `HFSafeTensorLoader::tensor(...) -> Tensor`.
- Tensor ownership: `torch::empty` creates owning tensors for model buffers and temporary runtime inputs. `torch::from_blob`/`make_tensor_from_blob` creates non-owning views, so the backing memory must outlive the returned `Tensor`.
- Memory management: owning polymorphic/resources are usually `std::unique_ptr` (`EngineCore`, `Scheduler`, `ModelRunner`, owned models, HF loader, `KVCache`, tokenizer impl). Non-owning dependencies are raw pointers (`Model*`, `KVCache*`, `Tokenizer*`, `ExecutionContext*`, `StackAllocator*`) and are validated before use.
- Allocator convention: `StackAllocator` is a monotonic per-step workspace; tensors from it are non-owning and valid only until reset/begin-step. `BlockAllocator` manages persistent fixed-size KV blocks; `KVCache::end_sequence()` must release them. In CUDA builds, `StackAllocator(size)` and the 3-argument `BlockAllocator` still default to CPU; pass `ParallelConfig::cuda(device_id)` explicitly for CUDA memory.
- Device dispatch convention: CUDA builds must dispatch by tensor device, not by compile flag alone. CPU tensors in a CUDA build must remain on CPU code paths. This is required for CPU default runtime, CTest compatibility, and mixed CPU/CUDA diagnostic tools.
- KV cache sizing: LLaMA runtime tools/tests must size `kv_block_size_bytes` from the HF config as `2 * block_size_tokens * (num_key_value_heads * head_dim) * sizeof(float)`. Tiny placeholder sizes such as 256 bytes are invalid for SmolLM2/Llama because runtime attention writes real K/V data into the blocks.
- Paged attention metadata: `RuntimeContext` carries `slot_mapping[B]`, `seq_indices[B]`, `context_lens[num_seqs]`, and `block_tables[num_layers, num_seqs, max_blocks_per_seq]`. `seq_indices` maps each scheduled row back to its sequence row in `block_tables`; LLaMA attention uses `layer_id` plus this sequence index to find physical KV blocks. The operator still exposes scoped metadata plumbing internally, but new model code should pass metadata through `RuntimeContext`, not global setters.
- Scheduler generation accounting: when prefill finishes, the model output sampled from the final prefill row is the first generated token and must be appended to the request. Decode steps then append one sampled token per scheduled row. Be careful not to drop the prefill sample or double-increment `num_computed`.
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

Configure CUDA against a Python PyTorch/libtorch install and explicit CUDA toolkit:

```bash
cmake -S . -B build-cuda \
  -DTINYLLM_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-cuda -j
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

When using the local SmolLM2 model for runtime checks:

```bash
TINYLLM_HF_TINY_LLAMA_DIR=~/models/smollm2-135M \
ctest --test-dir build --output-on-failure
```

Run CUDA CTest with SmolLM2 smoke tests:

```bash
TINYLLM_HF_TINY_LLAMA_DIR=~/models/smollm2-135M \
LD_LIBRARY_PATH=/path/to/torch/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
ctest --test-dir build-cuda --output-on-failure
```

CUDA builds register `test_llama_generation_cuda_smoke`, which runs `tools/llama_engine_generate --device cuda:0` and checks deterministic SmolLM2 token IDs for `hello` and `tiny llm inference`.

Run the currently approved tiny runtime test binary directly:

```bash
./build/tests/test_tiny_lm_runtime
```

Run the SmolLM2 runtime smoke test with the local model path:

```bash
./build/tests/test_tiny_lm_runtime ~/models/smollm2-135M
```

Run examples:

```bash
./build/tiny_lm_inference
./build/llama_inference
```

# Debug Tools & Alignment Scripts (调试工具)

The C++ files in `tools/` are standalone debugging runtimes built by `tests/CMakeLists.txt`; they are not themselves CTest test sources. The Python files in `scripts/` compare tool output against PyTorch/Transformers or inspect model files.

- `tools/hf_safetensor_dump.cpp`: inspect selected safetensors weights and model metadata. Build with `cmake --build build -j`, then run `./build/tools/hf_safetensor_dump ~/models/smollm2-135M`.
- `tools/llama_logits_dump.cpp`: dump C++ logits for explicit token IDs into a binary file for script-based comparison. Normally use it through `scripts/compare_llama_logits_with_transformers.py`.
- `tools/llama_tensor_dump.cpp`: dump C++ intermediate tensors such as embedding, per-layer norms, QKV, attention outputs, MLP activations, final norm, and logits. Normally use it through `scripts/compare_llama_tensors_with_transformers.py`.
- `tools/llama_engine_generate.cpp`: run `LLMEngine` greedy generation for one or more prompts and emit JSON lines containing output text and generated token IDs. It supports `--device cpu`, `--device cuda`, and `--device cuda:<device_id>`. Normally use it through `scripts/compare_llama_generation_with_transformers.py` or `scripts/run_llama_generation_smoke.py`.

Historical `test_llama_phase*` files were temporary bring-up checks for the LLaMA integration and should not be restored as regular tests. Use the focused runtime smoke test plus the Transformers comparison scripts for ongoing coverage.

Compare the custom safetensors loader with PyTorch safetensors:

```bash
python3 scripts/compare_hf_safetensor_with_pytorch.py \
  --dump-binary build/tools/hf_safetensor_dump \
  --model-dir ~/models/smollm2-135M
```

Compare C++ logits with Transformers:

```bash
TINYLLM_HF_TINY_LLAMA_DIR=~/models/smollm2-135M \
python3 scripts/compare_llama_logits_with_transformers.py \
  --dump-binary build/tools/llama_logits_dump
```

Compare intermediate C++ tensors with Transformers:

```bash
python3 scripts/compare_llama_tensors_with_transformers.py \
  --dump-binary build/tools/llama_tensor_dump \
  --model-dir ~/models/smollm2-135M \
  --tokens 1 22172 318
```

Compare full greedy generation between `LLMEngine` and Transformers:

```bash
python3 scripts/compare_llama_generation_with_transformers.py \
  --engine-binary build/tools/llama_engine_generate \
  --model-dir ~/models/smollm2-135M \
  --max-new-tokens 8 \
  --prompt hello \
  --prompt 'tiny llm inference' \
  --show-only
```

Run greedy generation smoke directly on CUDA:

```bash
./build-cuda/tools/llama_engine_generate \
  --device cuda:0 \
  ~/models/smollm2-135M \
  8 \
  hello \
  'tiny llm inference'
```

# Gotchas & Landmines (避坑指南)

[TODO: Human developer please fill in memory management and FFI landmines here]
