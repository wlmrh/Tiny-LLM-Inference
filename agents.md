# Project Context (项目上下文)

Tiny-LLM-Inference is a small vLLM-inspired single-process inference engine with a frontend/core split: `LLMEngine` handles text/tokenizer I/O, while `EngineCore` runs scheduler + model execution over token IDs. The HF runtime now covers LLaMA/SmolLM2-compatible models and Qwen2-family checkpoints such as Qwen2.5-1.5B-Instruct. The runtime uses FCFS continuous batching semantics across `running` and `waiting` queues, supports chunked prefill, one-token decode steps, simplified tail preemption, and paged KV-cache metadata through `SchedulerOutput`.

# Tech Stack & Tooling (技术栈与工具)

- Language standard: C++17, configured in the root `CMakeLists.txt` via `set(CMAKE_CXX_STANDARD 17)`.
- Build system: CMake 3.18+, with a single static library target `tiny_llm` and compatibility aliases `tiny_llm_core`, `tiny_llm_models`, and `tiny_llm_operators`.
- libtorch integration: `find_package(Torch QUIET)` first, then Python `torch.utils.cmake_prefix_path`, then `$HOME/libtorch*`. `${TORCH_LIBRARIES}` is linked into `tiny_llm`; `TORCH_CXX_FLAGS` is appended globally so the libtorch ABI flag is inherited by the project and third-party tokenizer code.
- CPU/CUDA mode: CPU is the default (`TINYLLM_ENABLE_CUDA=OFF`) and compiles `src/core/cpu/cpu_allocator.cpp`. CUDA mode (`TINYLLM_ENABLE_CUDA=ON`) enables CUDA language, requires `CUDAToolkit`, links `CUDA::cudart`, `CUDA::cublas`, `${TORCH_LIBRARIES}`, and compiles CUDA allocator/operator kernels. CUDA builds must still keep CPU as the default runtime unless `EngineArgs::parallel_config` or a tool option explicitly selects CUDA.
- Tokenizer integration: the Rust-backed HuggingFace tokenizer path is provided by `tokenizers-cpp`, fetched with CMake `FetchContent` from `https://github.com/mlc-ai/tokenizers-cpp.git` and linked as `tokenizers_cpp`. CMake requires `cargo` before fetching it.
- FetchContent update policy: after the initial tokenizers-cpp and googletest populations, CMake should not update/fetch them on every configure. Keep `FETCHCONTENT_UPDATES_DISCONNECTED` enabled so repeated local/remote configures do not depend on remote availability.
- Test framework: C++ tests use GoogleTest, pulled by `tests/CMakeLists.txt` with CMake `FetchContent` and linked through `GTest::gtest_main`. GoogleTest is built from source in the build tree so it inherits libtorch's global ABI flag; do not link a prebuilt system/Conda GTest that may use a different `_GLIBCXX_USE_CXX11_ABI` value.
- Tokenizer wrapper path: C++ runtime wrappers live in `include/tiny_llm/runtime/tokenizer.h` and `src/runtime/tokenizer.cpp`. `HFLlamaTokenizer` loads `tokenizer.json` via `tokenizers::Tokenizer::FromBlobJSON` or `tokenizer.model` via `FromBlobSentencePiece`. HF tokenizer configs may encode special tokens as strings, objects with `content`, or JSON `null`; Qwen2 checkpoints may omit `unk_token_id` and may report tokenizer vocab smaller than the padded model vocab.

# Architecture (架构与目录)

- Test/tool layout: automated GoogleTest/CTest entry points live under `tests/unit/` and `tests/integration/`; standalone C++ debugging runtimes live under `tools/`; Python comparison/debug helpers live under `scripts/`. Keep new diagnostic executables out of `tests/` unless they are the actual test entry point. `tests/CMakeLists.txt` uses `gtest_discover_tests()` for C++ test cases, while Python Transformers/smoke checks remain direct CTest registrations.
- Runtime test coverage: scheduler/KV/engine behavior is covered by `tests/unit/test_scheduler.cpp`, `tests/unit/test_kv_cache_manager.cpp`, `tests/unit/test_model_runner_prepared_inputs.cpp`, and `tests/unit/test_engine_core.cpp`. These tests use fake models or small host-side KV pools rather than real HF weights, so they should stay fast and deterministic. The old redundant `test_model_blocks.cpp` smoke test was removed; add focused module/operator coverage instead of restoring it.
- Runtime scheduling: `include/tiny_llm/runtime/scheduler.h` and `src/runtime/scheduler.cpp`. `Scheduler` owns request state, `waiting`/`running` queues, token budgets, chunked prefill/decode selection, preemption, and the runtime `KVCache` used by generation. `KVCacheManager` bridges scheduling decisions to KV block allocation and can return all per-layer block tables for scheduled requests.
- Engine frontend/core: `include/tiny_llm/runtime/engine.h`, `src/runtime/engine.cpp`, `include/tiny_llm/runtime/engine_core.h`, and `src/runtime/engine_core.cpp`. `LLMEngine` converts user text to core requests; `EngineCore::step()` calls `Scheduler::schedule()`, `ModelRunner::run()`, then `Scheduler::update_from_output()`. `EngineCore` passes the scheduler-owned `KVCache*` into `ModelRunner`; do not create a second runner-local KV cache for the same engine.
- Model execution: `include/tiny_llm/runtime/model_runner.h` and `src/runtime/model_runner.cpp`. `ModelRunner` loads HF weights from either `model.safetensors` or all `*.safetensors` shards in sorted order, then flattens scheduled request tokens into `PreparedInputs` (`input_ids`, `positions`, `slot_mapping`, `seq_indices`, `context_lens`, rank-3 `block_tables`, and `sample_row_offsets`), creates an explicit `RuntimeContext`, runs `Model::forward(const PreparedInputs&, RuntimeContext&)`, then greedily samples the request-final rows. The legacy `ModelExecutor`/`forward_step()` path has been removed.
- Model layer split: public model APIs live under `include/tiny_llm/models/`. `Model` is the causal-LM runtime interface; `LlamaForCausalLM` owns `LlamaModel` plus the LM-head projection; `LlamaModel` owns embeddings, decoder layers, and final norm. Qwen2-family support currently reuses this LLaMA-style runtime path rather than introducing a separate Qwen model class. Reusable building blocks live under `include/tiny_llm/models/modules/` and currently include `Embedding`, `Linear`, `RMSNorm`, and `RotaryEmbedding`. Keep request lifecycle, batch layout, and sampling out of model modules.
- Attention: public API in `include/tiny_llm/operators/paged_attention.h`; implementation in `src/operators/paged_attention/paged_attention.cpp`; CUDA baseline kernel in `src/operators/paged_attention/paged_attention_kernels.cu`. LLaMA/Qwen2 self-attention orchestration is in `include/tiny_llm/models/llama_decoder_layer.h` and `src/models/llama_layer.cpp`; Qwen2 uses the same GQA path plus optional q/k/v projection bias. LLaMA attention reads paged metadata from `RuntimeContext::attention_metadata()`, uses a scoped `PagedAttentionRuntimeMetadataGuard` while calling the existing operator, and with `ExecutionContext::kv()` writes/reads persistent KV blocks for runtime prefill/decode. Without KV metadata it falls back to in-batch causal attention for direct model tools such as `llama_tensor_dump`.
- CUDA paged attention: the current CUDA `llama_attention` path is a correctness bridge. It accepts CUDA tensors, copies small control metadata to CPU for validation/indexing, wraps KV blocks with `torch::from_blob(... device=kv_cache.device())`, writes current K/V into paged KV cache, gathers context K/V by `block_tables`/`seq_indices`/positions, and computes attention via libtorch `matmul + softmax + matmul`. Do not treat this as the final optimized paged-attention kernel.
- KV Cache memory pool: metadata service in `include/tiny_llm/runtime/kv_cache.h` and `src/runtime/kv_cache.cpp`; physical block allocator in `include/tiny_llm/core/allocator.h` and `src/core/allocator_common.cpp`, with CPU/CUDA backend files under `src/core/cpu/` and `src/core/cuda/`. CPU LLaMA KV data is float32 and stored per physical block as `[K block][V block]`, where each side is `block_size_tokens * kv_hidden_size` floats and `kv_hidden_size = num_key_value_heads * head_dim`.
- Model weight loading: HuggingFace config loader in `include/tiny_llm/models/hf_llama_config_loader.h` and `src/models/hf_llama_config_loader.cpp`; safetensors loader in `include/tiny_llm/models/hf_safetensors_loader.h` and `src/models/hf_safetensors_loader.cpp`; name-to-weight registry in `include/tiny_llm/models/llama_weight_map.h` and `src/models/llama_weight_map.cpp`; LLaMA/Qwen2 model wiring in `include/tiny_llm/models/llama_model.h` and `src/models/llama_model.cpp`. Do not assume the HF checkpoint is a single `model.safetensors` file; sharded `*.safetensors` directories are valid.
- C++/Rust wrapper layer: `src/runtime/tokenizer.cpp` conditionally includes `<tokenizers_cpp.h>`, defines a fallback C++ declaration for the external `tokenizers::Tokenizer` API, and wraps the handle inside `HFLlamaTokenizer::Impl`.
- Public headers: current public API is under `include/tiny_llm/`. Some legacy-looking headers exist under `include/core/` and `include/models/`; prefer `include/tiny_llm/...` for new code.

# Coding Conventions (代码规范)

- Tensor type: `tiny_llm::Tensor` is currently `using Tensor = torch::Tensor` in `include/tiny_llm/core/tensor.h`. Helper functions such as `tensor_dtype`, `tensor_shape`, `tensor_data`, and `make_tensor_from_blob` should be preferred over reaching through custom legacy Tensor assumptions.
- Tensor passing: read-only tensor inputs are generally passed as `const Tensor&`; mutable outputs and scratch buffers are passed as `Tensor&`; factory/load functions may return `Tensor` by value because `torch::Tensor` is an intrusive reference-counted handle. Examples: `Model::forward(const PreparedInputs& inputs, RuntimeContext& ctx) -> Tensor`, `ops::attention_paged(const Tensor& q, Tensor& out, ...)`, and `HFSafeTensorLoader::tensor(...) -> Tensor`.
- Tensor ownership: `torch::empty` creates owning tensors for model buffers and temporary runtime inputs. `torch::from_blob`/`make_tensor_from_blob` creates non-owning views, so the backing memory must outlive the returned `Tensor`.
- Memory management: owning polymorphic/resources are usually `std::unique_ptr` (`EngineCore`, `Scheduler`, `ModelRunner`, owned models, HF loader, `KVCache`, tokenizer impl). Non-owning dependencies are raw pointers (`Model*`, `KVCache*`, `Tokenizer*`, `ExecutionContext*`, `StackAllocator*`) and are validated before use.
- Allocator convention: `StackAllocator` is a monotonic per-step workspace; tensors from it are non-owning and valid only until reset/begin-step. `BlockAllocator` manages persistent fixed-size KV blocks; `KVCache::end_sequence()` must release them. In CUDA builds, `StackAllocator(size)` and the 3-argument `BlockAllocator` still default to CPU; pass `ParallelConfig::cuda(device_id)` explicitly for CUDA memory.
- Device dispatch convention: CUDA builds must dispatch by tensor device, not by compile flag alone. CPU tensors in a CUDA build must remain on CPU code paths. This is required for CPU default runtime, CTest compatibility, and mixed CPU/CUDA diagnostic tools.
- KV cache sizing: LLaMA/Qwen2 runtime tools/tests must size `kv_block_size_bytes` from the HF config as `2 * block_size_tokens * (num_key_value_heads * head_dim) * sizeof(float)`. Tiny placeholder sizes such as 256 bytes are invalid for SmolLM2/Llama because runtime attention writes real K/V data into the blocks.
- Paged attention metadata: `RuntimeContext` carries `slot_mapping[B]`, `seq_indices[B]`, `context_lens[num_seqs]`, and `block_tables[num_layers, num_seqs, max_blocks_per_seq]`. `seq_indices` maps each scheduled row back to its sequence row in `block_tables`; LLaMA attention uses `layer_id` plus this sequence index to find physical KV blocks. The operator still exposes scoped metadata plumbing internally, but new model code should pass metadata through `RuntimeContext`, not global setters.
- Scheduler generation accounting: when prefill finishes, the model output sampled from the final prefill row is the first generated token and must be appended to the request. Decode steps then append one sampled token per scheduled row. Be careful not to drop the prefill sample or double-increment `num_computed`.
- Error handling: the C++ code favors fail-fast `std::runtime_error` with contextual prefixes for invalid configuration, shape/dtype mismatch, missing files, invalid token IDs, allocator exhaustion, and tokenizer/FFI construction failures. Scheduler preemption cleanup is best-effort in a narrow path and catches exceptions to preserve forward progress.
- FFI/tokenizer handling: `HFLlamaTokenizer` stores the external tokenizer handle in a move-only pImpl (`std::unique_ptr<Impl>`). `from_model_dir()` validates model directory, tokenizer file presence, handle construction, vocab size, and special token IDs, then throws `std::runtime_error` on failure. Treat `unk_token_id` as optional and keep tokenizer validity aligned with model `vocab_size` for padded vocab checkpoints such as Qwen2.5.

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

Run an individual GoogleTest binary directly, optionally filtering cases:

```bash
./build/tests/test_scheduler --gtest_filter='SchedulerTest.*'
./build/tests/test_tiny_lm_runtime ~/models/smollm2-135M --gtest_filter='TinyLMRuntimeIntegrationTest.*'
```

Run the SmolLM2 runtime integration tests through CTest:

```bash
TINYLLM_HF_TINY_LLAMA_DIR=~/models/smollm2-135M \
ctest --test-dir build --output-on-failure \
  -R 'TinyLMRuntimeIntegrationTest|LLMOfflineIntegrationTest|test_llama_generation_cpu_smoke'
```

Run examples:

```bash
./build/tiny_lm_inference
./build/llama_inference
```


Run the Qwen2.5-1.5B-Instruct CUDA smoke test:

```bash
./build-cuda/tools/llama_engine_generate \
  --device cuda:0 \
  /models/Qwen2.5-1.5B-Instruct \
  8 \
  hello \
  '你好'
```

Run focused CUDA regression tests after Qwen/LLaMA runtime changes:

```bash
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure \
  -R 'Scheduler|KVCache|ModelRunner|EngineCore|PagedAttention|test_llama_generation_cuda_smoke'
```


# Debug Tools & Alignment Scripts (调试工具)

The C++ files in `tools/` are standalone debugging runtimes built by `tests/CMakeLists.txt`; they are not themselves CTest test sources. The Python files in `scripts/` compare tool output against PyTorch/Transformers or inspect model files.

- `tools/hf_safetensor_dump.cpp`: inspect selected safetensors weights and model metadata. Build with `cmake --build build -j`, then run `./build/tools/hf_safetensor_dump ~/models/smollm2-135M`.
- `tools/llama_logits_dump.cpp`: dump C++ logits for explicit token IDs into a binary file for script-based comparison. Normally use it through `scripts/compare_llama_logits_with_transformers.py`.
- `tools/llama_tensor_dump.cpp`: dump C++ intermediate tensors such as embedding, per-layer norms, QKV, attention outputs, MLP activations, final norm, and logits. Normally use it through `scripts/compare_llama_tensors_with_transformers.py`.
- `tools/llama_engine_generate.cpp`: run `LLMEngine` greedy generation for one or more prompts and emit JSON lines containing output text and generated token IDs. It supports `--device cpu`, `--device cuda`, and `--device cuda:<device_id>`. Normally use it through `scripts/compare_llama_generation_with_transformers.py` or `scripts/run_llama_generation_smoke.py`.
- `tools/llama_engine_benchmark.cpp`: run an end-to-end offline LLM benchmark using the real `LLM` path. It supports `--device`, `--warmup`, `--repeat`, `--max-new-tokens`, repeated `--prompt`, and `--json`; it reports load/init time, first-token latency, total latency, prompt/generated token counts, and throughput. Build it via CMake and run it manually; do not register benchmark runs as regular CTest tests.

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

Run the end-to-end benchmark manually:

```bash
./build/tools/llama_engine_benchmark \
  --warmup 1 \
  --repeat 3 \
  --max-new-tokens 8 \
  --json \
  ~/models/smollm2-135M
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

Qwen2.5-specific notes:

- Use `/models/Qwen2.5-1.5B-Instruct` as the default server model path. On AutoDL-style hosts, keep the real files under `/root/autodl-tmp/models/` and expose `/models/...` as a symlink to avoid filling the 30G system disk.
- Qwen2.5-1.5B-Instruct uses tied embeddings, GQA (`num_attention_heads=12`, `num_key_value_heads=2`), q/k/v projection bias, `rope_theta=1000000`, and no `unk_token_id`. Do not reintroduce assumptions that every HF tokenizer has object-form special token entries or a valid unknown token.
- Qwen2.5 `config.json` may have JSON `null` for optional token IDs. Optional integer readers should treat null as missing, not as a type error.
- Direct Hugging Face access may be unavailable on rented servers. `hf-mirror.com` or ModelScope can be used to fetch the required files; validated files are `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, and `model.safetensors`.
