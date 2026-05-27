# Tiny-LLM-Inference

Tiny-LLM-Inference is a compact C++17 decoder-only LLM inference engine inspired by vLLM. It is mainly used to validate runtime pieces such as scheduling, paged KV cache, model execution, and greedy generation for small LLaMA/SmolLM2/Qwen2-family checkpoints.

Current validated local models on this server:

```text
/models/smollm2-135M
/models/Qwen2.5-1.5B-Instruct
```

## Layout

```text
include/tiny_llm/   Public headers
src/                Runtime, models, operators, and CPU/CUDA backends
tools/              Manual debug/generation tools
benchmark/          Manual performance benchmarks
tests/              GoogleTest and CTest correctness tests
scripts/            Python comparison helpers
assets/tiny_lm/     Tiny toy-model assets
```

## Dependencies

Required:

- CMake 3.18+
- C++17 compiler
- Python 3
- PyTorch/libtorch
- Rust `cargo` for `tokenizers-cpp`

Optional:

- CUDA Toolkit when building with `TINYLLM_ENABLE_CUDA=ON`
- Python packages for Transformers comparison and baseline benchmarks: `torch`, `transformers`, `safetensors`

CMake discovers libtorch through `find_package(Torch)`, Python PyTorch's `torch.utils.cmake_prefix_path`, or `$HOME/libtorch*`.

## Build

CPU build:

```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=OFF
cmake --build build -j
```

CUDA build:

```bash
cmake -S . -B build-cuda -DTINYLLM_ENABLE_CUDA=ON
cmake --build build-cuda -j
```

If CMake cannot find PyTorch/libtorch, pass the Python PyTorch prefix explicitly:

```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=OFF \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j
```

## Run Generation

CPU SmolLM2:

```bash
./build/tools/llama_engine_generate \
  /models/smollm2-135M \
  8 \
  hello \
  "tiny llm inference"
```

CUDA SmolLM2:

```bash
./build-cuda/tools/llama_engine_generate \
  --device cuda:0 \
  /models/smollm2-135M \
  8 \
  hello \
  "tiny llm inference"
```

CUDA Qwen2.5 smoke:

```bash
./build-cuda/tools/llama_engine_generate \
  --device cuda:0 \
  /models/Qwen2.5-1.5B-Instruct \
  8 \
  hello \
  "你好"
```

Each generation command prints one JSON object per prompt.

## Benchmark

TinyLLM CPU benchmark:

```bash
./build/benchmark/llama_engine_benchmark \
  --warmup 1 \
  --repeat 3 \
  --max-new-tokens 8 \
  --json \
  /models/smollm2-135M
```

TinyLLM CUDA benchmark, default CUDA path:

```bash
./build-cuda/benchmark/llama_engine_benchmark \
  --device cuda:0 \
  --warmup 1 \
  --repeat 3 \
  --max-new-tokens 8 \
  --json \
  /models/smollm2-135M
```

TinyLLM CUDA benchmark with the custom paged-attention kernel enabled:

```bash
TINYLLM_PAGED_ATTENTION_BACKEND=cuda \
./build-cuda/benchmark/llama_engine_benchmark \
  --device cuda:0 \
  --warmup 1 \
  --repeat 3 \
  --max-new-tokens 8 \
  --json \
  /models/smollm2-135M
```

Transformers baseline:

```bash
python3 benchmark/transformers_generate_benchmark.py \
  --warmup 1 \
  --repeat 3 \
  --max-new-tokens 8 \
  --json \
  /models/smollm2-135M
```

TinyLLM vs Transformers comparison:

```bash
python3 benchmark/run_benchmark_comparison.py \
  --tinyllm-binary build/benchmark/llama_engine_benchmark \
  --warmup 1 \
  --repeat 3 \
  --max-new-tokens 8 \
  --json \
  /models/smollm2-135M
```

CUDA comparison:

```bash
python3 benchmark/run_benchmark_comparison.py \
  --tinyllm-binary build-cuda/benchmark/llama_engine_benchmark \
  --device cuda:0 \
  --warmup 1 \
  --repeat 3 \
  --max-new-tokens 8 \
  --json \
  /models/smollm2-135M
```

Benchmark output includes readable latency/token/throughput sections and a final JSON line when `--json` is used. TinyLLM additionally reports runtime breakdown fields such as `prepare_inputs_ms`, `prefill_ms`, `decode_ms_total`, `decode_ms_per_token`, and `sampling_ms`.


### Industrial Offline Serving Benchmark

`benchmark/industrial_benchmark.py` runs a reproducible offline-serving workload suite on top of the existing TinyLLM and Transformers benchmark entry points. It generates deterministic synthetic prompts with the Hugging Face tokenizer so each scenario targets real token counts instead of approximate character lengths.

Default target on the remote RTX 4090 server:

```bash
python3 benchmark/industrial_benchmark.py \
  --backend tinyllm \
  --warmup 1 \
  --repeat 3 \
  --model-dir /models/Qwen2.5-1.5B-Instruct \
  --tinyllm-binary build-cuda/benchmark/llama_engine_benchmark \
  --device cuda:0
```

The standard suite contains these workloads:

| scenario | batch | target input tokens per prompt | max new tokens |
| --- | ---: | ---: | ---: |
| `interactive` | 1 | 128 | 64 |
| `chat_serving` | 8 | 128 | 128 |
| `long_prefill` | 4 | 1024 | 64 |
| `decode_heavy` | 4 | 256 | 256 |
| `throughput` | 16 | 128 | 128 |

Run a quick validation pass before a long benchmark:

```bash
python3 benchmark/industrial_benchmark.py \
  --quick \
  --backend all \
  --warmup 0 \
  --repeat 1 \
  --model-dir /models/Qwen2.5-1.5B-Instruct \
  --tinyllm-binary build-cuda/benchmark/llama_engine_benchmark \
  --device cuda:0 \
  --label quick_comparison
```

Run TinyLLM for the full workload matrix and only compare Transformers on selected representative scenarios:

```bash
python3 benchmark/industrial_benchmark.py \
  --backend all \
  --transformers-scenarios interactive,chat_serving,long_prefill \
  --warmup 1 \
  --repeat 3 \
  --model-dir /models/Qwen2.5-1.5B-Instruct \
  --tinyllm-binary build-cuda/benchmark/llama_engine_benchmark \
  --device cuda:0 \
  --label qwen25_1p5b_cuda4090
```

If Transformers is too slow for large batch or long decode scenarios, use `--backend tinyllm` for the full matrix and run a separate smaller `--backend all --scenarios interactive` comparison. Reports are written to `benchmark/results/` as both JSON and Markdown files. The Markdown report includes model, GPU, batch size, prompt tokens, generated tokens, TTFT, total latency, decode ms/token, end-to-end tokens/s, decode tokens/s, repeat data, and TinyLLM/Transformers ratios when both backends are present.

Current validation status on the remote server:

- `python3 -m py_compile benchmark/industrial_benchmark.py` passed.
- `cmake --build build-cuda -j` passed.
- Existing TinyLLM CUDA smoke benchmark passed for `/models/Qwen2.5-1.5B-Instruct`.
- `--quick --backend tinyllm --warmup 0 --repeat 1` generated JSON and Markdown reports.
- `--quick --backend all --warmup 0 --repeat 1` generated TinyLLM and Transformers comparison reports.
- TinyLLM and Transformers benchmark entry points now load each model once per benchmark process; warmup/repeat iterations measure steady-state generation.
- A previous full `--backend all` run was intentionally stopped during `chat_serving` before this steady-state benchmark fix.

This benchmark is offline batched generation, not a request-rate serving benchmark. It does not report p50/p95/p99 request latency or QPS under arrival-rate control; those require a separate server/load-generator harness.

## Test

Run the default CPU correctness tests:

```bash
ctest --test-dir build --output-on-failure
```

Run focused runtime tests:

```bash
ctest --test-dir build --output-on-failure \
  -R 'Scheduler|KVCache|ModelRunner|EngineCore|PagedAttention'
```

Run model-backed tests with SmolLM2:

```bash
TINYLLM_HF_TINY_LLAMA_DIR=/models/smollm2-135M \
ctest --test-dir build --output-on-failure
```

Run the runtime integration binary directly:

```bash
./build/tests/test_tiny_lm_runtime /models/smollm2-135M
```

Run CUDA tests after a CUDA build:

```bash
TINYLLM_HF_TINY_LLAMA_DIR=/models/smollm2-135M \
ctest --test-dir build-cuda --output-on-failure
```

Some model-backed tests skip when the model path or Python dependencies are unavailable.

## Alignment Helpers

Compare generation with Transformers:

```bash
python3 scripts/compare_llama_generation_with_transformers.py \
  --engine-binary build/tools/llama_engine_generate \
  --model-dir /models/smollm2-135M \
  --max-new-tokens 8 \
  --prompt hello \
  --prompt "tiny llm inference"
```

Compare logits:

```bash
TINYLLM_HF_TINY_LLAMA_DIR=/models/smollm2-135M \
python3 scripts/compare_llama_logits_with_transformers.py \
  --dump-binary build/tools/llama_logits_dump
```

Compare intermediate tensors:

```bash
python3 scripts/compare_llama_tensors_with_transformers.py \
  --dump-binary build/tools/llama_tensor_dump \
  --model-dir /models/smollm2-135M \
  --tokens 1 22172 318
```

Inspect safetensors:

```bash
python3 scripts/compare_hf_safetensor_with_pytorch.py \
  --dump-binary build/tools/hf_safetensor_dump \
  --model-dir /models/smollm2-135M
```

## Notes

- Correctness tests live in `tests/`; benchmarks live in `benchmark/` and are not registered as CTest tests.
- CUDA paged attention currently has both a default reference path and an opt-in custom kernel path selected by `TINYLLM_PAGED_ATTENTION_BACKEND=cuda`.
- Detailed architecture notes and coding conventions are maintained in `agents.md`.
