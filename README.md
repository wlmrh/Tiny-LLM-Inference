# Tiny-LLM-Inference

[![CI](https://github.com/wlmrh/Tiny-LLM-Inference/actions/workflows/ci.yml/badge.svg?event=pull_request)](https://github.com/wlmrh/Tiny-LLM-Inference/actions/workflows/ci.yml?query=event%3Apull_request)
[![Release v0.1.0](docs/release-badge.svg)](https://github.com/wlmrh/Tiny-LLM-Inference/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Tiny-LLM-Inference is a compact C++17/CUDA decoder-only LLM inference engine inspired by vLLM. It implements the core path from Hugging Face checkpoint loading through request scheduling, paged KV cache management, model execution, sampling, and reproducible correctness/performance evaluation.

The project is intentionally scoped as a **single-process, single-device, offline runtime** for systems learning and controlled experiments. It is not an HTTP/gRPC service, distributed serving stack, or production-SLA claim.

## Architecture

![Tiny-LLM-Inference architecture](docs/architecture.svg)

Nested boxes show ownership, rounded rectangles are executable components, and cylinders are KV-storage views. Dashed blue arrows show the `EngineCore`-mediated request/result flow between stages; the return from `ModelRunner` is applied by `Scheduler::update_from_output()` before an `EngineCoreOutput` reaches output processing. Green arrows show KV management. The two `Paged KV cache` cylinders are aliases of the same scheduler-owned storage, repeated only to keep the diagram's routing orthogonal. Data types are written on the producing or consuming component rather than drawn as peer modules.

- `LLM` owns the deployment resources and `LLMEngine`; `LLMEngine` owns input/output processing and the token-level `EngineCore`.
- `InputPreprocessor` produces validated `EngineCoreRequest` objects, while `OutPreprocessor` turns `EngineCoreOutput` into incremental `UserOutput`.
- `EngineCore` drives each step by calling `Scheduler::schedule()`, `ModelRunner::run()`, and then `Scheduler::update_from_output()`.
- `Scheduler` owns request state, queues, token budgets, preemption, and `KVCacheManager`; the model reads and writes the same scheduler-owned paged KV cache during attention.
- `Sampler` returns token IDs inside `ModelRunnerOutput`; `EngineCore` passes that output to `Scheduler`, so the sampler and scheduler do not communicate directly.
- LLaMA/SmolLM2 and Qwen2-family checkpoints share the LLaMA-style model path.

See [Architecture](docs/Architecture.md) for ownership, scheduling, KV cache, tensor metadata, and device boundaries.

## Highlights

- Hugging Face `tokenizer.json`/SentencePiece and single-file or sharded safetensors loading.
- LLaMA/SmolLM2-compatible and Qwen2-family checkpoints, including Qwen2.5-1.5B-Instruct.
- FCFS-style waiting/running queues with chunked prefill/decode and tail preemption.
- Scheduler-owned paged KV cache with explicit runtime attention metadata.
- CPU execution and an optional single-device CUDA path with experimental BF16 compute/KV modes.
- Transformers/vLLM comparison, request-event tracing, and offline/open-loop benchmark suites.

## Verified v0.1.0 Results

Release validation used candidate commit `25e2355921b033abb89091d15f768c57c715c63c` with `git_dirty=false`, an NVIDIA GeForce RTX 4080 SUPER, driver 595.71.05, CUDA 12.8, pinned Qwen2.5-1.5B-Instruct revision `989aa798...`, and FP32 compute/KV for headline measurements.

Hardware disclosure: the cloud benchmark host reported the device name `NVIDIA GeForce RTX 4080 SUPER`
and `32760 MiB` of memory. [NVIDIA's retail reference specification](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080-family/)
lists 16 GB GDDR6X. The mismatch may reflect cloud-platform device presentation or nonstandard provisioning,
but its exact cause was not independently verified. These results are tied to the provider-exposed benchmark
host and should not be treated as measurements of a standard retail 16 GB card solely from the GPU label.

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

TinyLLM outperformed the tested Transformers baseline across the three performance workloads, but trailed vLLM in end-to-end and decode throughput. Long-prefill is the clearest limitation: TinyLLM TTFT was 520.538 ms versus vLLM's 62.298 ms. See the [complete v0.1.0 benchmark report](benchmark/reports/v0.1.0/README.md) for workload definitions, all open-loop percentiles, raw JSON, pinned model hashes, and reproduction commands.

Performance results apply only to their recorded environment and workload. A baseline comparison is not a claim of production parity with vLLM, SGLang, or TensorRT-LLM.

## Realistic-v1 Workload Results

The realistic-v1 experiment keeps the engine unchanged and replaces uniform synthetic prompts with three
non-overlapping 1,000-request BurstGPT timing/length windows plus length-matched OASST1 conversation prompts.
It uses Qwen2.5-1.5B-Instruct, FP32 compute/KV, greedy decoding, and runtime candidate `272b8f0` on the same
provider-exposed RTX 4080 SUPER environment described by the hardware disclosure above.

| Validation | Result |
| --- | --- |
| Three-backend EOS-aware correctness | 8/8 prompts; exact pairwise token-ID match |
| Reference-capacity calibrations | 3/3 completed |
| Trace replays | 12/12 completed; 1,000/1,000 requests each; zero reported errors |
| Published evidence | Per-request/grouped metrics, source/model hashes, sanitized selection metadata, and raw-artifact checksum |

| Offline cohort | TinyLLM E2E tok/s | Transformers | vLLM | TinyLLM / Transformers | TinyLLM / vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short chat | 393.798 | 281.178 | 705.089 | 1.401x | 0.559x |
| Medium chat | 307.089 | 255.891 | 706.412 | 1.200x | 0.435x |
| Long prefill | 35.185 | 98.758 | 345.995 | 0.356x | 0.102x |
| Long decode | 331.814 | 270.578 | 720.053 | 1.226x | 0.461x |

| Replay load | Median TTFT p99 ms | Median E2E p99 ms | Median relative good-request ratio |
| --- | ---: | ---: | ---: |
| 0.25C_ref | 18,830.780 | 22,874.800 | 0.998 |
| 0.50C_ref | 19,645.290 | 35,652.500 | 0.981 |
| 0.75C_ref | 54,901.300 | 55,878.116 | 0.847 |
| 0.90C_ref | 57,359.100 | 70,610.760 | 0.716 |

TinyLLM exceeded the tested Transformers path in short-chat, medium-chat, and long-decode median throughput,
but long-prefill remained a clear weakness and vLLM led all four performance cohorts. The decline in relative
goodput above 0.50C_ref shows increasing queueing pressure even though every request completed. `C_ref` is an
experiment-local simultaneous-batch completion rate, not a production saturation capacity or SLA. See the
[complete realistic-v1 report](benchmark/reports/realistic-v1/README.md) for window composition, min/median/max
trace statistics, per-window results, methodology, limitations, JSON artifacts, and reproduction evidence.

## Three-Step Reproduction

```bash
# 1. Configure and build the CPU release preset.
cmake --preset cpu-release
cmake --build --preset cpu-release -j

# 2. Run deterministic local generation.
./build/cpu-release/offline_llm /models/smollm2-135M cpu

# 3. Check benchmark command plumbing without running a long benchmark.
python3 benchmark/industrial_benchmark.py --preset quick --dry-run
```

## Status and Scope

- Local Hugging Face checkpoint directories only; no model download layer is included in the runtime.
- CPU is the default backend; CUDA supports one selected device when enabled at build time.
- No concurrent multi-engine guarantee, serving protocol, tensor/pipeline parallelism, prefix caching, LoRA, quantization, or multimodal path.
- BF16 is an experimental CUDA path with FP32 master weights/operator boundaries; benchmark it on the target workload rather than assuming a speedup.

See [Project Status](docs/Project_Status.md) for the support matrix, explicit non-goals, and benchmark-claim policy.

## C++ API Example

```cpp
#include "tiny_llm/runtime/llm.h"

#include <iostream>
#include <vector>

int main()
{
    tiny_llm::LLM llm("/models/smollm2-135M");

    tiny_llm::LLMSamplingParams params;
    params.temperature = 0.8f;
    params.top_p = 0.95f;

    const std::vector<std::string> prompts = {
        "Hello, my name is",
        "The capital of France is",
    };

    const auto outputs = llm.generate(prompts, params);
    for (const auto &output : outputs)
    {
        std::cout << output.text << "\n";
    }
}
```

The same flow is available through the built example:

```bash
./build/cpu-release/offline_llm /models/smollm2-135M cpu
./build/cuda-release/offline_llm /models/Qwen2.5-1.5B-Instruct cuda:0
```

## Requirements

- CMake 3.20+
- C++17 compiler
- Python with PyTorch/libtorch available to CMake
- Rust `cargo` for `tokenizers-cpp`
- CUDA Toolkit when building with `TINYLLM_ENABLE_CUDA=ON`

## Build

Preset-based builds are the recommended reproducible path:

```bash
cmake --preset cpu-release
cmake --build --preset cpu-release -j
```

CUDA build:

```bash
cmake --preset cuda-release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
cmake --build --preset cuda-release -j
```

The explicit configure path remains available for custom build directories:

```bash
cmake -S . -B build-custom -DTINYLLM_ENABLE_CUDA=OFF
cmake --build build-custom -j
```

Main outputs are the `tiny_llm` static library, the `offline_llm` API example, generation/debug tools under the selected build directory, and tests under its `tests/` subdirectory.

## Run Generation

Use a local Hugging Face checkpoint directory containing `config.json`, `tokenizer.json` or `tokenizer.model`, and safetensors weights.

```bash
./build/cpu-release/tools/llama_engine_generate \
  /models/smollm2-135M \
  8 \
  hello
```

```bash
./build/cuda-release/tools/llama_engine_generate \
  --device cuda:0 \
  --dtype bf16 \
  --kv-cache-dtype bf16 \
  /models/Qwen2.5-1.5B-Instruct \
  8 \
  hello
```

`llama_engine_generate` prints one JSON object per prompt and also supports `--kv-num-blocks N`.

## Benchmarks

The config-driven suite writes workload JSONL, TinyLLM request event traces, summary JSON, and Markdown reports:

```bash
python3 benchmark/run_benchmark_suite.py \
  --config benchmark/configs/qwen25_quick.json \
  --backend tinyllm,transformers
```

For focused optimization loops and release regression checks:

```bash
python3 benchmark/industrial_benchmark.py --preset focus
python3 benchmark/industrial_benchmark.py --preset regression
```

See [Tools, Tests, and Benchmarks](docs/modules/Tools_Tests_and_Benchmarks.md) for workload and reporting details.
The [realistic-v1 benchmark report](benchmark/reports/realistic-v1/README.md) complements rather than replaces
the verified v0.1.0 release baseline above.

## Tests

```bash
cmake --preset ci-cpu
cmake --build --preset ci-cpu --parallel 2
ctest --preset ci-cpu
```

Model-backed tests use `TINYLLM_HF_TINY_LLAMA_DIR`; they skip rather than downloading a model. Public CI runs model-independent CPU tests. CUDA and real-model checks are release gates on a GPU host.

## Documentation

Start with [docs/README.md](docs/README.md). Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md), releases are recorded in [CHANGELOG.md](CHANGELOG.md), and the source is available under the [Apache License 2.0](LICENSE).
