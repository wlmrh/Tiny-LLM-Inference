# Tiny-LLM-Inference Agent Guide

## Project Context

Tiny-LLM-Inference is a small vLLM-inspired, single-process inference engine. `LLMEngine` owns text/tokenizer I/O; `EngineCore` runs scheduling and model execution over token IDs. The HF runtime supports LLaMA/SmolLM2-compatible models and Qwen2-family checkpoints such as Qwen2.5-1.5B-Instruct.

Default project workflow: perform code changes, builds, tests, benchmarks, and other project commands on the session-provided remote server unless the user explicitly scopes a task to local files. Use the SSH command and password supplied for the current session instead of assuming a persistent host alias.

## Stack And Build Assumptions

- C++17, CMake 3.18+, one static library target `tiny_llm` plus compatibility aliases `tiny_llm_core`, `tiny_llm_models`, and `tiny_llm_operators`.
- libtorch is discovered by `find_package(Torch QUIET)`, Python `torch.utils.cmake_prefix_path`, then `$HOME/libtorch*`. Preserve global `TORCH_CXX_FLAGS` propagation so third-party code inherits the libtorch ABI flag.
- CPU is the default runtime (`TINYLLM_ENABLE_CUDA=OFF`). CUDA builds enable CUDA language, require `CUDAToolkit`, link CUDA runtime/cuBLAS/libtorch, but must still dispatch CPU tensors through CPU paths unless CUDA is explicitly selected by config or tool option.
- Tokenizer support comes from Rust-backed `tokenizers-cpp` via CMake `FetchContent`; CMake requires `cargo`.
- Keep `FETCHCONTENT_UPDATES_DISCONNECTED` enabled after first population of tokenizers-cpp and googletest.
- C++ tests use GoogleTest fetched in `tests/CMakeLists.txt`; do not link a prebuilt system/Conda GTest with a potentially mismatched `_GLIBCXX_USE_CXX11_ABI`.

## Repository Layout

- Public headers: prefer `include/tiny_llm/...`; avoid adding new code under legacy-looking `include/core/` or `include/models/`.
- Runtime: `include/tiny_llm/runtime/`, `src/runtime/`.
- Models: `include/tiny_llm/models/`, `src/models/`.
- Operators: `include/tiny_llm/operators/`, `src/operators/`.
- Automated tests: `tests/unit/`, `tests/integration/`, registered through `tests/CMakeLists.txt`.
- Standalone debug tools: `tools/`.
- Python comparison/debug scripts: `scripts/`.
- Manual benchmarks and benchmark wrappers: `benchmark/`; do not register benchmark runs as regular CTest tests.

## Runtime Architecture

- `Scheduler` owns request state, `waiting`/`running` queues, token budgets, chunked prefill/decode selection, simplified tail preemption, and the runtime `KVCache`.
- `EngineCore::step()` calls `Scheduler::schedule()`, `ModelRunner::run()`, then `Scheduler::update_from_output()`. `EngineCore` passes the scheduler-owned `KVCache*` into `ModelRunner`; do not create a second runner-local KV cache for the same engine.
- `ModelRunner` loads either `model.safetensors` or all sorted `*.safetensors` shards, prepares `PreparedInputs`, builds `RuntimeContext`, calls `Model::forward(const PreparedInputs&, RuntimeContext&)`, and greedily samples request-final rows.
- LLaMA and Qwen2 reuse the LLaMA-style runtime path. `LlamaForCausalLM` owns `LlamaModel` plus LM head; model modules stay limited to reusable layers such as `Embedding`, `Linear`, `RMSNorm`, and `RotaryEmbedding`.
- Attention reads paged metadata from `RuntimeContext::attention_metadata()`. New model code should pass metadata through `RuntimeContext`, not through global setters.
- CUDA paged attention is currently a correctness bridge using libtorch operations and some CPU-side metadata handling; do not treat it as the final optimized kernel.

## Coding Rules

- `tiny_llm::Tensor` is `torch::Tensor`; prefer helper APIs such as `tensor_dtype`, `tensor_shape`, `tensor_data`, and `make_tensor_from_blob`.
- Pass read-only tensors as `const Tensor&`, mutable outputs/scratch as `Tensor&`, and return tensor handles by value when loading or creating tensors.
- `torch::empty` owns memory. `torch::from_blob`/`make_tensor_from_blob` are non-owning views; backing storage must outlive the tensor.
- Use `std::unique_ptr` for owned polymorphic/resources and raw pointers for validated non-owning dependencies.
- `StackAllocator` is per-step monotonic workspace; tensors from it are invalid after reset/begin-step. `BlockAllocator` owns persistent fixed-size KV blocks; `KVCache::end_sequence()` must release them.
- In CUDA builds, default allocator constructors still create CPU memory. Pass `ParallelConfig::cuda(device_id)` explicitly for CUDA memory.
- Dispatch by tensor device, not compile flag alone. CPU tensors in CUDA builds must remain valid on CPU paths.
- Runtime KV block bytes for LLaMA/Qwen2 must be `2 * block_size_tokens * (num_key_value_heads * head_dim) * sizeof(float)`.
- `RuntimeContext` carries `slot_mapping[B]`, `seq_indices[B]`, `context_lens[num_seqs]`, and rank-3 `block_tables[num_layers, num_seqs, max_blocks_per_seq]`.
- When prefill finishes, the sample from the final prefill row is the first generated token. Do not drop it or double-increment `num_computed`.
- Prefer fail-fast `std::runtime_error` with contextual prefixes for invalid config, shapes, dtype, files, token IDs, allocator exhaustion, and tokenizer/FFI failures.

## Tokenizer And HF Model Notes

- Runtime tokenizer wrappers live in `include/tiny_llm/runtime/tokenizer.h` and `src/runtime/tokenizer.cpp`.
- `HFLlamaTokenizer` loads `tokenizer.json` with `tokenizers::Tokenizer::FromBlobJSON` or `tokenizer.model` with `FromBlobSentencePiece`.
- HF tokenizer configs may encode special tokens as strings, objects with `content`, or JSON `null`.
- Qwen2 checkpoints may omit `unk_token_id` and may report tokenizer vocab smaller than padded model vocab; treat unknown token IDs as optional.
- HF checkpoints may be sharded; never assume a single `model.safetensors`.

## Build And Test Commands

CPU build:

```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA build:

```bash
cmake -S . -B build-cuda \
  -DTINYLLM_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-cuda -j
```

Focused runtime checks:

```bash
TINYLLM_HF_TINY_LLAMA_DIR=~/models/smollm2-135M \
ctest --test-dir build --output-on-failure \
  -R 'TinyLMRuntimeIntegrationTest|LLMOfflineIntegrationTest|test_llama_generation_cpu_smoke'

ctest --test-dir build-cuda --output-on-failure \
  -R 'Scheduler|KVCache|ModelRunner|EngineCore|PagedAttention|test_llama_generation_cuda_smoke'
```

Useful direct runs:

```bash
./build/tests/test_scheduler --gtest_filter='SchedulerTest.*'
./build/tests/test_tiny_lm_runtime ~/models/smollm2-135M --gtest_filter='TinyLMRuntimeIntegrationTest.*'

./build-cuda/tools/llama_engine_generate \
  --device cuda:0 \
  /models/Qwen2.5-1.5B-Instruct \
  8 \
  hello \
  '你好'
```

## Benchmark Policy

- Do not run full industrial benchmarks after every small change.
- Use `python3 benchmark/industrial_benchmark.py --preset focus` for tight optimization loops.
- Use `--preset regression` before claiming a coherent optimization improved batch/decode behavior.
- Use `--preset full` only at phase boundaries or before reporting headline numbers.
- Use `--preset quick --dry-run` for command plumbing checks.
- Use `--preset profile_prefill --profile-detail` only for bottleneck diagnosis; synchronized per-component timings are not headline throughput numbers.
- `benchmark/results/` keeps only recent reports; preserve a report elsewhere only when the user asks.

## Debug And Alignment Tools

- `tools/hf_safetensor_dump.cpp`: inspect safetensors weights and metadata.
- `tools/llama_logits_dump.cpp`: compare C++ logits through `scripts/compare_llama_logits_with_transformers.py`.
- `tools/llama_tensor_dump.cpp`: compare intermediate tensors through `scripts/compare_llama_tensors_with_transformers.py`.
- `tools/llama_engine_generate.cpp`: run greedy generation and JSONL output; also used by generation comparison and smoke scripts.
- `benchmark/llama_engine_benchmark.cpp`: end-to-end TinyLLM benchmark binary.
- `benchmark/transformers_generate_benchmark.py`: Hugging Face Transformers baseline.
- `benchmark/run_benchmark_comparison.py`: TinyLLM/Transformers benchmark comparison wrapper.

Historical `test_llama_phase*` files were temporary bring-up checks and should not be restored as regular tests.

## Qwen2.5 Landmines

- Default server model path: `/models/Qwen2.5-1.5B-Instruct`. On AutoDL-style hosts, keep real files under `/root/autodl-tmp/models/` and expose `/models/...` as a symlink to avoid filling the system disk.
- Qwen2.5-1.5B-Instruct uses tied embeddings, GQA (`num_attention_heads=12`, `num_key_value_heads=2`), q/k/v projection bias, `rope_theta=1000000`, and no `unk_token_id`.
- Optional integer readers must treat JSON `null` as missing, not as a type error.
- Direct Hugging Face access may be unavailable on rented servers; `hf-mirror.com` or ModelScope can be used. Validated files are `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, and `model.safetensors`.
