# Tiny-LLM-Inference

Tiny-LLM-Inference is a compact C++17 decoder-only LLM inference engine inspired by vLLM. It focuses on the core runtime pieces needed for offline generation: request scheduling, paged KV cache management, Hugging Face weight/tokenizer loading, model execution, default greedy decoding, and seeded temperature/top-k/top-p sampling.

The current runtime targets LLaMA-style checkpoints, including small LLaMA/SmolLM2-compatible models and Qwen2-family models such as Qwen2.5-1.5B-Instruct.

## Project Layout

```text
include/tiny_llm/   Public C++ headers
src/                Runtime, model, operator, allocator, and CPU/CUDA code
tests/              GoogleTest unit and integration tests
tools/              Debug, alignment, and generation executables
scripts/            Python comparison and smoke-check helpers
benchmark/          Manual benchmark binaries and wrappers
docs/               Architecture and module-level documentation
examples/           Small API examples
```

## Requirements

- CMake 3.18+
- C++17 compiler
- Python 3
- PyTorch/libtorch
- Rust `cargo` for `tokenizers-cpp`
- CUDA Toolkit when building with `TINYLLM_ENABLE_CUDA=ON`
- Optional Python packages for Transformers comparison: `transformers`, `safetensors`

CMake looks for libtorch with `find_package(Torch)`, then Python PyTorch's `torch.utils.cmake_prefix_path`, then `$HOME/libtorch*`. The first configure may fetch `tokenizers-cpp` and GoogleTest through CMake `FetchContent`.

## Build

CPU build:

```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=OFF
cmake --build build -j
```

CUDA build:

```bash
cmake -S . -B build-cuda \
  -DTINYLLM_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-cuda -j
```

Main build outputs:

- `tiny_llm`: static library for runtime, models, operators, and core support.
- `offline_llm`: small `LLM` API example.
- `build/tools/*`: debug and generation tools.
- `build/benchmark/*`: manual benchmark executables.
- `build/tests/*`: GoogleTest executables.

## Run Generation

Use any local Hugging Face checkpoint directory that contains `config.json`, a tokenizer file, and safetensors weights. The paths below are examples.

```bash
./build/tools/llama_engine_generate \
  /models/smollm2-135M \
  8 \
  hello \
  "tiny llm inference"
```

CUDA:

```bash
./build-cuda/tools/llama_engine_generate \
  --device cuda:0 \
  /models/Qwen2.5-1.5B-Instruct \
  8 \
  hello \
  "你好"
```

`llama_engine_generate` prints one JSON object per prompt and also supports `--kv-num-blocks N`. It runs deterministic `temperature=0` generation and applies `generation_config.json` repetition penalty when present.

The higher-level API example can be run with:

```bash
./build/offline_llm /models/smollm2-135M cpu
./build-cuda/offline_llm /models/Qwen2.5-1.5B-Instruct cuda:0
```

## Test

Default correctness tests:

```bash
ctest --test-dir build --output-on-failure
```

Focused runtime tests:

```bash
ctest --test-dir build --output-on-failure \
  -R 'Scheduler|KVCache|ModelRunner|EngineCore|PagedAttention'
```

Model-backed tests and Python comparison tests use `TINYLLM_HF_TINY_LLAMA_DIR` and skip when the model or Python dependencies are unavailable:

```bash
TINYLLM_HF_TINY_LLAMA_DIR=/models/smollm2-135M \
ctest --test-dir build --output-on-failure \
  -R 'LLMRuntimeIntegrationTest|LLMOfflineIntegrationTest|test_llama_generation_cpu_smoke'
```

CUDA tests use the CUDA build directory:

```bash
ctest --test-dir build-cuda --output-on-failure \
  -R 'Scheduler|KVCache|ModelRunner|EngineCore|PagedAttention|test_llama_generation_cuda_smoke'
```

## Benchmarks

For quick command plumbing checks:

```bash
python3 benchmark/industrial_benchmark.py --preset quick --dry-run
```

For focused optimization loops:

```bash
python3 benchmark/industrial_benchmark.py --preset focus
```

For pre-report regression checks:

```bash
python3 benchmark/industrial_benchmark.py --preset regression
```

Use `--preset full` only when a longer benchmark is intentional. Reports are written under `benchmark/results/` unless `--output-dir` is overridden.

The benchmark wrapper defaults to the CUDA Qwen2.5 path used by the project workflow. Override `--model-dir`, `--tinyllm-binary`, and `--device` for other environments.

The lower-level benchmark binary is also available:

```bash
./build-cuda/benchmark/llama_engine_benchmark \
  --device cuda:0 \
  --warmup 1 \
  --repeat 3 \
  --max-new-tokens 8 \
  --json \
  /models/Qwen2.5-1.5B-Instruct
```

## Debug And Alignment

Common helper tools:

- `llama_logits_dump`: dump final logits for Transformers comparison.
- `llama_tensor_dump`: dump intermediate tensors for alignment.
- `hf_safetensor_dump`: inspect safetensors keys, shapes, dtypes, and metadata.
- `scripts/compare_llama_generation_with_transformers.py`: compare TinyLLM generation with Transformers.
- `scripts/run_llama_generation_smoke.py`: run model-backed generation smoke checks.

## Documentation

Start with [docs/README.md](docs/README.md) for architecture and module documentation. Repository-specific agent workflow and coding conventions are in [AGENTS.md](AGENTS.md).
