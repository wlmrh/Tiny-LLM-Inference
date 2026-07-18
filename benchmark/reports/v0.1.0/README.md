# Tiny-LLM-Inference v0.1.0 Benchmark Report

## 1. Executive Summary

Tiny-LLM-Inference v0.1.0 passed its CPU, CUDA, real-model, deterministic-generation, and benchmark release gates on an NVIDIA GeForce RTX 4080 SUPER. With Qwen2.5-1.5B-Instruct in greedy FP32 mode, TinyLLM, Transformers, and vLLM produced exactly matching token IDs for every correctness request. No backend was skipped.

The results show a clear, workload-dependent profile. TinyLLM was faster than the tested Transformers baseline in all three performance workloads. Against vLLM, TinyLLM had lower interactive TTFT (18.716 ms versus 32.426 ms), but lower end-to-end and decode throughput. The largest gap was long-prefill: TinyLLM TTFT was 520.538 ms versus vLLM's 62.298 ms. These numbers are controlled offline measurements, not evidence of production-serving parity.

Five open-loop workloads derived from a measured capacity of 3.932141 requests/s completed 200/200 requests each. Every request event stream contained one submit, one admit, 64 ordered token events, and one finish; all recorded latencies were nonnegative and all reported percentiles were monotonic.

## 2. System Scope and Claim Boundary

Tiny-LLM-Inference is a single-process, single-device, offline decoder-only inference runtime. It contains scheduling, paged KV-cache ownership, model execution, and incremental request events, but no HTTP/gRPC endpoint, production router, distributed serving layer, multi-GPU execution, or SLA mechanism.

The comparison answers a narrow question: how this release candidate behaves on the recorded hardware, software, model, dtype, and workloads. It does not claim production readiness, state-of-the-art performance, or equivalence to vLLM, SGLang, or TensorRT-LLM.

## 3. Hardware and Software Environment

| Item | Value |
| --- | --- |
| Runtime candidate | `25e2355921b033abb89091d15f768c57c715c63c` |
| Git worktree | `git_dirty=false` |
| GPU | NVIDIA GeForce RTX 4080 SUPER, 32760 MiB |
| Driver | 595.71.05 |
| CUDA Toolkit | 12.8 |
| OS | Linux 5.15.0-78-generic x86_64 |
| Suite Python | 3.12.3 |
| PyTorch / Transformers / safetensors | 2.8.0+cu128 / 4.55.2 / 0.8.0 |
| vLLM environment | vLLM 0.25.1, PyTorch 2.11.0+cu130, Transformers 5.14.1 |

Release validation completed with 12/12 Python benchmark unit tests, 65/65 CPU model-independent CTests, 7/7 model-backed CPU CTests, and 78/78 CUDA CTests with the real-model path enabled. Qwen2.5 FP32 generation, BF16 compute/KV smoke generation, and two-prompt Transformers token alignment also passed.

## 4. Model, Dtype, and Sampling Configuration

| Model | Official repository | Pinned revision | Weight SHA-256 |
| --- | --- | --- | --- |
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1` |
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |

The official Hugging Face revisions were transferred through `hf-mirror.com`, then loaded entirely from local files. SmolLM2 validation observed `model_type=llama`, tokenizer size 49152, and 272 safetensors keys. Qwen2.5 validation observed `model_type=qwen2`, tokenizer size 151665, and 338 keys. File-level sizes and SHA-256 values are recorded in [manifest.json](manifest.json).

Headline measurements use TinyLLM FP32 compute with FP32 KV cache, Transformers FP32, vLLM `float32`, and greedy decoding. Correctness retains EOS behavior. Performance and open-loop scenarios use fixed output with EOS ignored. Offline performance uses one warmup and three measured repeats; open-loop uses one warmup and one measured run because each run contains 200 requests.

BF16 is a correctness gate only. It is not included in the performance tables.

## 5. Correctness Methodology

The correctness workload uses batch 4, target ISL 32, target OSL 16, greedy sampling, and EOS enabled. The suite compares generated token ID arrays pairwise, rather than relying only on decoded text.

| Comparison | Exact token match | Mismatches |
| --- | --- | ---: |
| TinyLLM vs Transformers | Yes | 0 |
| TinyLLM vs vLLM | Yes | 0 |
| Transformers vs vLLM | Yes | 0 |

Each backend generated the configured 64 aggregate tokens. The three fixed-output performance workloads also produced their configured token counts on all backends, with no errors or skips; their outputs happened to match exactly as well, although output equality is not used as a performance-ratio requirement.

## 6. Offline Workloads

| Workload | Batch | Target ISL | Target OSL | Aggregate generated tokens |
| --- | ---: | ---: | ---: | ---: |
| interactive | 1 | 128 | 64 | 64 |
| long-prefill | 4 | 1024 | 64 | 256 |
| decode-heavy | 8 | 128 | 128 | 1024 |

All workloads use simultaneous offline arrival. TTFT is time to the first generated token. E2E throughput divides aggregate generated tokens by total measured latency. Decode throughput excludes the first-token interval according to the benchmark's existing metric definition.

## 7. Offline Results

| Workload | Backend | TTFT ms | Total latency ms | E2E tok/s | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| interactive | TinyLLM | 18.716 | 727.400 | 87.985 | 88.897 |
| interactive | Transformers | 26.268 | 1406.450 | 45.505 | 45.646 |
| interactive | vLLM | 32.426 | 620.530 | 103.138 | 107.124 |
| long-prefill | TinyLLM | 520.538 | 1947.332 | 131.462 | 176.620 |
| long-prefill | Transformers | 514.774 | 2154.655 | 118.813 | 153.670 |
| long-prefill | vLLM | 62.298 | 716.604 | 357.240 | 385.141 |
| decode-heavy | TinyLLM | 109.237 | 2034.515 | 503.314 | 527.716 |
| decode-heavy | Transformers | 117.006 | 3586.784 | 285.493 | 292.814 |
| decode-heavy | vLLM | 55.157 | 1399.754 | 731.557 | 755.617 |

The source-of-truth data, including repeat values, commands, samples, ratios, and TinyLLM request summaries, is [offline.json](offline.json).

## 8. Open-loop Capacity Definition

Capacity is derived only from the release run's TinyLLM decode-heavy result:

```text
batch = 8
avg_total_latency_ms = 2034.515
C = batch * 1000 / avg_total_latency_ms
  = 8 * 1000 / 2034.515
  = 3.932141075391 requests/s
```

The suite command uses `--capacity-rps 3.9321`. No additional safety factor and no historical capacity value were used.

Open-loop workloads use Qwen2.5-1.5B-Instruct, TinyLLM FP32 compute/FP32 KV, ISL 128, fixed OSL 64, and 200 requests. Four workloads use seeded Poisson arrivals at 0.25C, 0.50C, 0.75C, and 0.90C. One workload uses fixed-interval arrivals at 0.50C.

## 9. Open-loop Results

| Workload | Rate req/s | Completed | Experiment wall-clock s | Queue p99 ms | TTFT p50 / p95 / p99 ms | Engine TTFT p99 ms | E2E p50 / p95 / p99 ms | TPOT p99 ms |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |
| Poisson 0.25C | 0.983025 | 200/200 | 217.276 | 28.605 | 32.000 / 44.030 / 60.802 | 34.000 | 812.550 / 888.860 / 903.052 | 13.810 |
| Poisson 0.50C | 1.966050 | 200/200 | 109.008 | 30.903 | 37.805 / 57.440 / 65.606 | 56.901 | 863.800 / 924.235 / 956.106 | 14.350 |
| Poisson 0.75C | 2.949075 | 200/200 | 72.924 | 26.723 | 39.410 / 59.815 / 65.634 | 56.506 | 892.900 / 994.865 / 1022.414 | 15.424 |
| Poisson 0.90C | 3.538890 | 200/200 | 60.889 | 28.425 | 40.375 / 61.240 / 66.929 | 56.605 | 916.400 / 1033.010 / 1068.512 | 16.259 |
| Fixed 0.50C | 1.966050 | 200/200 | 102.009 | 11.500 | 37.225 / 43.305 / 44.200 | 33.002 | 842.600 / 851.205 / 853.202 | 12.867 |

Experiment wall-clock is the duration of the 200-request run, including the arrival schedule. It must not be read as per-request latency. Per-request queue, TTFT, engine TTFT, E2E, and TPOT percentiles come from submit/admit/token/finish timestamps. The complete source data is [open_loop.json](open_loop.json).

For all five workloads, the independent release check found 200 request IDs and 13,400 events: 200 submit, 200 admit, 12,800 token, and 200 finish events. Token indices were exactly 0 through 63 per request. There were no duplicate/missing lifecycle events, negative times, incomplete summaries, or percentile-order violations.

## 10. Interpretation

TinyLLM's strongest result is the interactive first-token path: its 18.716 ms TTFT is lower than both measured baselines, and its total latency is roughly half the Transformers baseline. Decode-heavy batching also delivers 1.76x the Transformers E2E throughput and 1.80x its decode throughput.

The main limitation is comparison with vLLM. TinyLLM reaches 85.3% of vLLM's interactive E2E throughput and 68.8% of its decode-heavy E2E throughput. Long-prefill is the weakest case: TinyLLM reaches 36.8% of vLLM's E2E throughput, while its TTFT is 8.36x higher. The release therefore demonstrates a coherent learning/research runtime with reproducible measurements, not a replacement for a mature serving engine.

Open-loop tail latency increases gradually with Poisson load: E2E p99 rises from 903.052 ms at 0.25C to 1068.512 ms at 0.90C. Fixed 0.50C is less bursty than Poisson 0.50C and correspondingly has lower queue, TTFT, E2E, and TPOT tails. These runs remain well below the measured offline token-throughput ceiling because each request emits 64 tokens and arrivals are intentionally spread over wall-clock time.

## 11. Limitations

- Single process and one CUDA device only.
- No HTTP/gRPC serving layer, network latency, request router, or production admission control.
- One GPU, one primary benchmark model, FP32 headline dtype, greedy decoding, and a small workload matrix.
- vLLM uses a separate Python environment, so package stacks are recorded rather than assumed identical.
- Load/init time is reported in raw JSON but excluded from throughput interpretation because process lifecycle choices differ across backends.
- TinyLLM allocator memory fields describe allocator-managed tensors; they are not process-total GPU memory and are not compared as a fair memory ratio against Transformers or vLLM.
- BF16 retains FP32 master weights/operator boundaries and is reported only as a correctness smoke gate.

## 12. Reproduction Commands

Configure the release environment and build:

```bash
source /root/autodl-tmp/tinyllm-env.sh

cmake --preset ci-cpu
cmake --build --preset ci-cpu --parallel 2
ctest --preset ci-cpu

cmake --preset cuda-release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
cmake --build --preset cuda-release --parallel 2
ctest --preset cuda-release
```

Run the regression, offline, and open-loop suites:

```bash
python3 benchmark/industrial_benchmark.py \
  --preset regression \
  --model-dir /models/Qwen2.5-1.5B-Instruct \
  --tinyllm-binary build/cuda-release/benchmark/llama_engine_benchmark \
  --device cuda:0 \
  --output-dir benchmark/results/v0.1.0-regression

python3 benchmark/run_benchmark_suite.py \
  --config benchmark/configs/qwen25_week1_offline.json \
  --model-dir /models/Qwen2.5-1.5B-Instruct \
  --tinyllm-binary build/cuda-release/benchmark/llama_engine_benchmark \
  --vllm-python /root/autodl-tmp/venvs/vllm/bin/python \
  --backend tinyllm,transformers,vllm \
  --output-dir benchmark/results/v0.1.0-offline \
  --label v0.1.0-offline

python3 benchmark/run_benchmark_suite.py \
  --config benchmark/configs/qwen25_week1_open_loop.json \
  --model-dir /models/Qwen2.5-1.5B-Instruct \
  --tinyllm-binary build/cuda-release/benchmark/llama_engine_benchmark \
  --backend tinyllm \
  --capacity-rps 3.9321 \
  --output-dir benchmark/results/v0.1.0-open-loop \
  --label v0.1.0-open-loop
```

## 13. Raw Artifact Manifest

- [offline.json](offline.json): exact suite JSON, SHA-256 `48a542956163d142601b8420d98e7bc6a87a0b705ce8742666cbeb8b8d7230bb`
- [open_loop.json](open_loop.json): exact suite JSON, SHA-256 `831463dbbf3c861d883072d7513cf617f4d808d7c3006e6702d7521b759b3914`
- [manifest.json](manifest.json): environment, model file hashes, validation gates, commands, and artifact metadata
- Release asset: `tinyllm-v0.1.0-benchmark-artifacts.tar.gz`, 1,045,512 bytes, SHA-256 `6dd368008271799d1c4b67e12c89b81eabee369d68ed3927434117585b604a86`
- Checksum asset: `tinyllm-v0.1.0-benchmark-artifacts.sha256`

The release tarball contains the full regression, offline, and open-loop run directories, including generated Markdown/JSON, workload JSONL, event JSONL, the server-side model manifest, and validation logs. The tarball and checksum are GitHub Release assets and are intentionally not committed to the repository.
