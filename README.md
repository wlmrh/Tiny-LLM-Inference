# Tiny-LLM-Inference

[![CI](https://github.com/wlmrh/Tiny-LLM-Inference/actions/workflows/ci.yml/badge.svg?event=pull_request)](https://github.com/wlmrh/Tiny-LLM-Inference/actions/workflows/ci.yml?query=event%3Apull_request)
[![Release v0.1.0](docs/release-badge.svg)](https://github.com/wlmrh/Tiny-LLM-Inference/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Tiny-LLM-Inference is a compact C++17/CUDA decoder-only LLM inference engine inspired by vLLM. It covers the path from Hugging Face checkpoint loading to scheduling, paged KV cache management, model execution, sampling, and correctness/performance evaluation.

The project is intentionally scoped as a **single-process, single-device, offline runtime** for systems learning and controlled experiments. It is not a production serving stack.

## Highlights

- Loads local Hugging Face tokenizers and single-file or sharded safetensors checkpoints.
- Supports LLaMA/SmolLM2-compatible and Qwen2-family models.
- Implements FCFS scheduling, chunked prefill/decode, tail preemption, and scheduler-owned paged KV cache.
- Runs on CPU or one selected CUDA device, with experimental BF16 compute/KV modes.
- Provides correctness comparison, event tracing, and offline/open-loop benchmarks.

## Architecture

![Tiny-LLM-Inference architecture](docs/architecture.svg)

The runtime separates request processing, scheduling, model execution, and output handling:

- `LLM` owns deployment resources and `LLMEngine`; `LLMEngine` owns the input/output processors around the token-level `EngineCore`.
- Each engine step follows `Scheduler::schedule() -> ModelRunner::run() -> Scheduler::update_from_output()` under `EngineCore` coordination.
- `Scheduler` owns request state, queues, token budgets, preemption, and the paged KV cache lifecycle.
- `ModelRunner` prepares batched tensor metadata, executes the model, and samples the next tokens.
- Attention reads and writes the same scheduler-owned KV storage through block tables and slot mappings.

See [Architecture](docs/Architecture.md) for complete ownership, data flow, tensor metadata, and device boundaries.

## Requirements

- CMake 3.20+ and a C++17 compiler
- Python with PyTorch/libtorch discoverable by CMake
- Rust `cargo` for `tokenizers-cpp`
- CUDA Toolkit for CUDA builds

## Quick Start

Build the CPU preset:

```bash
cmake --preset cpu-release
cmake --build --preset cpu-release -j
```

Run generation with a local Hugging Face checkpoint containing `config.json`, tokenizer files, and safetensors weights:

```bash
./build/cpu-release/offline_llm /models/smollm2-135M cpu
```

CUDA build and execution:

```bash
cmake --preset cuda-release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
cmake --build --preset cuda-release -j

./build/cuda-release/offline_llm /models/Qwen2.5-1.5B-Instruct cuda:0
```

For custom build directories, configure with `-DTINYLLM_ENABLE_CUDA=ON|OFF`. The C++ API is documented in [Runtime API](docs/modules/Runtime_API.md).

## Tests

```bash
cmake --preset ci-cpu
cmake --build --preset ci-cpu --parallel 2
ctest --preset ci-cpu
```

Public CI runs model-independent CPU tests. Model-backed tests use `TINYLLM_HF_TINY_LLAMA_DIR` and skip when no checkpoint is provided. CUDA and real-model checks require a GPU host.

## Scope

- Local checkpoint directories only; the runtime does not download models.
- CPU is the default backend; CUDA is limited to one selected device.
- No HTTP/gRPC serving, distributed execution, tensor/pipeline parallelism, prefix caching, LoRA, quantization, or multimodal path.
- BF16 is experimental and retains FP32 master weights/operator boundaries.

See [Project Status](docs/Project_Status.md) for the support matrix, non-goals, and benchmark-claim policy.

## Performance

### v0.1.0 release baseline

Measured at commit `25e2355921b033abb89091d15f768c57c715c63c` with Qwen2.5-1.5B-Instruct (`989aa798...`), FP32 compute/KV, CUDA 12.8, driver 595.71.05, and a provider-exposed NVIDIA GeForce RTX 4080 SUPER.

| Validation | Result |
| --- | --- |
| TinyLLM / Transformers / vLLM greedy token agreement | Exact pairwise match; 0 mismatches, no backend skips |
| CPU and CUDA test suites | CPU 65/65 plus 7/7 model-backed; CUDA 78/78 |
| Five 200-request open-loop workloads | 200/200 each; complete events and monotonic percentiles |

| Workload | Backend | TTFT ms | E2E tok/s | Decode tok/s |
| --- | --- | ---: | ---: | ---: |
| interactive | TinyLLM | 18.716 | 87.985 | 88.897 |
| interactive | Transformers | 26.268 | 45.505 | 45.646 |
| interactive | vLLM | 32.426 | 103.138 | 107.124 |
| long-prefill | TinyLLM | 520.538 | 131.462 | 176.620 |
| long-prefill | Transformers | 514.774 | 118.813 | 153.670 |
| long-prefill | vLLM | 62.298 | 357.240 | 385.141 |
| decode-heavy | TinyLLM | 109.237 | 503.314 | 527.716 |
| decode-heavy | Transformers | 117.006 | 285.493 | 292.814 |
| decode-heavy | vLLM | 55.157 | 731.557 | 755.617 |

| Open-loop workload | TTFT p99 ms | E2E p99 ms |
| --- | ---: | ---: |
| Poisson 0.50C (1.966050 req/s) | 65.606 | 956.106 |
| Poisson 0.90C (3.538890 req/s) | 66.929 | 1068.512 |

TinyLLM delivered higher end-to-end and decode throughput than the tested Transformers baseline across these three workloads, but trailed vLLM on both metrics. Long-prefill was the clearest limitation. See the [complete v0.1.0 report](benchmark/reports/v0.1.0/README.md) for methodology, full percentiles, raw data, and reproduction commands.

### Realistic-v1 workloads

This experiment used three non-overlapping 1,000-request BurstGPT timing/length windows, length-matched OASST1 prompts, Qwen2.5-1.5B-Instruct, FP32 compute/KV, greedy decoding, and runtime candidate `c353eb4`.

| Validation | Result |
| --- | --- |
| Three-backend EOS-aware correctness | 8/8 prompts; exact pairwise token-ID match |
| Reference-capacity calibrations | 3/3 completed |
| Trace replays | 12/12 completed; 1,000/1,000 requests each; zero reported errors |

| Offline cohort | TinyLLM E2E tok/s | Transformers | vLLM | TinyLLM / Transformers | TinyLLM / vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short chat | 396.460 | 287.101 | 704.593 | 1.381x | 0.563x |
| Medium chat | 311.123 | 207.058 | 709.644 | 1.503x | 0.438x |
| Long prefill | 94.711 | 96.638 | 346.799 | 0.980x | 0.273x |
| Long decode | 324.153 | 242.411 | 720.375 | 1.337x | 0.450x |

| Replay load | Median TTFT p99 ms | Median E2E p99 ms | Median relative good-request ratio |
| --- | ---: | ---: | ---: |
| 0.25C_ref | 6,402.406 | 7,925.981 | 1.000 |
| 0.50C_ref | 6,634.000 | 21,967.294 | 1.000 |
| 0.75C_ref | 20,436.000 | 26,922.146 | 0.693 |
| 0.90C_ref | 26,008.320 | 35,076.000 | 0.617 |

TinyLLM delivered higher throughput than the tested Transformers path for short chat, medium chat, and long decode, while long prefill was approximately on par and vLLM led all four cohorts. `C_ref` is an experiment-local simultaneous-batch completion rate, not production capacity or an SLA. See the [complete realistic-v1 report](benchmark/reports/realistic-v1/README.md) for trace composition, per-window results, artifacts, and limitations.

All measurements apply only to their recorded environment and workload. The cloud host reported `32760 MiB` for a GPU identified as RTX 4080 SUPER, which differs from [NVIDIA's retail 16 GB specification](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080-family/); the cause was not independently verified. These comparisons are not claims of production parity.

### Reproduce benchmarks

```bash
python3 benchmark/industrial_benchmark.py --preset quick --dry-run

python3 benchmark/run_benchmark_suite.py \
  --config benchmark/configs/qwen25_quick.json \
  --backend tinyllm,transformers
```

Benchmark reports contain the full environment, model revision, workload definitions, raw artifacts, and reproduction commands.

## Documentation

Start with the [documentation index](docs/README.md). Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md), releases are recorded in [CHANGELOG.md](CHANGELOG.md), and the project is licensed under [Apache License 2.0](LICENSE).
