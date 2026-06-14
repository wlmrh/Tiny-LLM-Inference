# Tools, Tests, and Benchmarks Module

This module documents the project executables, test registration, and benchmark workflow.

## Main Files

- `CMakeLists.txt`
- `tests/CMakeLists.txt`
- `tools/*.cpp`
- `scripts/*.py`
- `benchmark/*.cpp`
- `benchmark/*.py`
- `examples/*.cpp`

## Build Targets

The top-level CMake build creates:

- `tiny_llm`: one static library containing runtime, models, operators, and core support.
- `tiny_llm_core`, `tiny_llm_models`, `tiny_llm_operators`: compatibility aliases.
- `offline_llm`: example executable.
- benchmark targets under `benchmark/`.
- test and debug-tool targets under `tests/`.

Torch is discovered first so `TORCH_CXX_FLAGS` can be propagated globally before third-party code builds. `tokenizers-cpp` is fetched with `FetchContent` and requires Rust `cargo`.

## Debug and Alignment Tools

- `hf_safetensor_dump`: inspect safetensor keys, shapes, dtype, and metadata.
- `llama_logits_dump`: dump final logits for Python/Transformers comparison.
- `llama_tensor_dump`: dump intermediate tensors for alignment.
- `llama_engine_generate`: run deterministic `temperature=0` generation and print JSONL outputs.

`llama_engine_generate` supports:

- `--device cpu|cuda[:id]`
- `--kv-num-blocks N`
- `<model_dir> <max_new_tokens> <prompt> [prompt...]`

It auto-estimates KV blocks when `--kv-num-blocks` is omitted.

## Tests

`tests/CMakeLists.txt` registers GoogleTest unit/integration tests through `gtest_discover_tests`.

Covered areas include:

- `ParallelConfig`
- runtime device config
- weight map
- module-level linear/RMSNorm behavior
- LLaMA helper ops
- paged attention CPU/CUDA
- sampler
- model runner prepared inputs
- scheduler
- KV cache manager
- engine core
- LLM runtime/offline integration

Python comparison/smoke tests are registered when Python is available. They use skip return code `77` when required models or dependencies are unavailable.

## Comparison Scripts

- `scripts/compare_llama_logits_with_transformers.py`
- `scripts/compare_llama_tensors_with_transformers.py`
- `scripts/compare_llama_generation_with_transformers.py`
- `scripts/run_llama_generation_smoke.py`

These scripts align TinyLLM behavior against HuggingFace Transformers or run model-backed smoke checks.

## Benchmarks

Benchmark files:

- `benchmark/llama_engine_benchmark.cpp`
- `benchmark/industrial_benchmark.py`
- `benchmark/transformers_generate_benchmark.py`
- `benchmark/vllm_generate_benchmark.py`
- `benchmark/run_benchmark_comparison.py`

Benchmark policy:

- Use focused presets for tight optimization loops.
- Use regression presets before claiming coherent throughput improvements.
- Use full presets only at phase boundaries or for headline reports.
- Keep benchmark outputs out of regular CTest registration.

## Common Commands

CPU build and test:

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

Generation smoke:

```bash
./build/tools/llama_engine_generate /models/smollm2-135M 8 hello
./build-cuda/tools/llama_engine_generate --device cuda:0 /models/smollm2-135M 8 hello
```
