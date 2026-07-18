# TinyLLM Realistic Workload Benchmark v1

## Executive summary

This benchmark combines BurstGPT arrival/length traces with length-matched OASST1 prompts.
It is a single-process, single-device internal submission benchmark, not a production HTTP SLA.

The trace replay uses three non-overlapping 1,000-request windows and calibrates each window
against its own simultaneous saturation throughput before replaying 0.25C through 0.90C.
The offline section compares identical prompt cohorts across TinyLLM, Transformers, and vLLM.

## Scope and claim boundary

Results apply only to the recorded commit, model, GPU, dtype, and workloads. Relative goodput
uses experiment-local thresholds and is not a production SLA or a claim of serving parity.

## Environment

- Runtime candidate commit: `272b8f0428abe0642ff72a79b5d9141a04964c53`
- git_dirty: `False`
- GPU / driver: `NVIDIA GeForce RTX 4080 SUPER, 595.71.05, 32760`
- CUDA toolkit: `Build cuda_12.8.r12.8/compiler.35583870_0`
- TinyLLM compute / KV dtype: `FP32 / FP32`
- Sampling: greedy; correctness retains EOS, performance uses fixed output

## Sources and workload construction

- OASST1 is a content proxy; it is not the original BurstGPT request text.
- BurstGPT token counts come from GPT services and are matched approximately under the Qwen tokenizer.
- Arrival timestamps are linearly scaled to workload-specific measured capacity.
- Fixed output lengths model requested inference work and ignore natural EOS behavior.
- Network, HTTP, authentication, multi-GPU, and production reliability overheads are excluded.

## Dataset manifest

- BurstGPT revision: `7eb2c4f8350f8a6985272386f5c14af1f678b299`
- BurstGPT SHA-256: `3326259f9efb11845bc5ef85fa97e6f691050b0974621c91ef22acd566c43a40`
- OASST1 revision: `fdf72ae0827c1cda404aff25b6603abec9e3399b`
- OASST1 SHA-256: `2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7`
- Qwen revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Windows: `3` × `1000` requests

## Trace capacity and replay results

`C` is measured independently for each window as `1000 × 1000 / generation_ms`.

| Window | Capacity req/s | Load | Target req/s | Completed | Req/s | TTFT p99 ms | TPOT p99 ms | E2E p99 ms | Max conc. | Good ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window-01 | 2.329551 | 0.25C | 0.582388 | 1000/1000 | 0.582774 | 18830.780 | 665.000 | 21130.000 | 50 | 1.0000 |
| window-01 | 2.329551 | 0.50C | 1.164776 | 1000/1000 | 1.159201 | 19645.290 | 729.580 | 24028.260 | 65 | 1.0000 |
| window-01 | 2.329551 | 0.75C | 1.747163 | 1000/1000 | 1.577297 | 61996.220 | 707.038 | 66959.000 | 170 | 0.6970 |
| window-01 | 2.329551 | 0.90C | 2.096596 | 1000/1000 | 1.748016 | 94571.670 | 707.345 | 99427.260 | 253 | 0.5510 |
| window-02 | 7.164920 | 0.25C | 1.791230 | 1000/1000 | 1.737281 | 21966.703 | 421.463 | 22874.800 | 218 | 0.9950 |
| window-02 | 7.164920 | 0.50C | 3.582460 | 1000/1000 | 3.241544 | 45207.500 | 653.500 | 46430.479 | 437 | 0.9630 |
| window-02 | 7.164920 | 0.75C | 5.373690 | 1000/1000 | 4.554024 | 54901.300 | 654.000 | 55878.116 | 521 | 0.8470 |
| window-02 | 7.164920 | 0.90C | 6.448428 | 1000/1000 | 5.263407 | 57359.100 | 654.000 | 58298.284 | 545 | 0.8270 |
| window-03 | 0.747389 | 0.25C | 0.186847 | 1000/1000 | 0.186750 | 6130.000 | 56.856 | 26213.380 | 7 | 0.9980 |
| window-03 | 0.747389 | 0.50C | 0.373694 | 1000/1000 | 0.372148 | 6221.140 | 94.427 | 35652.500 | 15 | 0.9810 |
| window-03 | 0.747389 | 0.75C | 0.560541 | 1000/1000 | 0.553596 | 14819.990 | 116.942 | 51749.300 | 27 | 0.8970 |
| window-03 | 0.747389 | 0.90C | 0.672650 | 1000/1000 | 0.659970 | 38969.310 | 126.524 | 70610.760 | 47 | 0.7160 |

The JSON report additionally contains p50/p95/p99 queue, TTFT, engine TTFT, TPOT, and E2E
for the overall population and every log-type, ISL, and OSL bucket, together with input/output/total
token throughput, wall-clock duration, and cross-window min/median/max summaries.

## Offline three-backend cohorts

### correctness

| Backend | E2E tok/s median | Decode tok/s median | Latency ms median |
| --- | ---: | ---: | ---: |
| tinyllm | 20.972 | 24.790 | 11348.558 |
| transformers | 38.707 | 110.804 | 6148.821 |
| vllm | 218.706 | 226.930 | 1088.218 |

### long_decode

| Backend | E2E tok/s median | Decode tok/s median | Latency ms median |
| --- | ---: | ---: | ---: |
| tinyllm | 331.814 | 354.843 | 6172.127 |
| transformers | 270.578 | 282.786 | 7568.969 |
| vllm | 720.053 | 735.929 | 2844.236 |

### long_prefill

| Backend | E2E tok/s median | Decode tok/s median | Latency ms median |
| --- | ---: | ---: | ---: |
| tinyllm | 35.185 | 134.498 | 7275.857 |
| transformers | 98.758 | 145.201 | 2592.183 |
| vllm | 345.995 | 376.758 | 739.894 |

### medium_chat

| Backend | E2E tok/s median | Decode tok/s median | Latency ms median |
| --- | ---: | ---: | ---: |
| tinyllm | 307.089 | 353.595 | 3334.536 |
| transformers | 255.891 | 301.308 | 4001.703 |
| vllm | 706.412 | 735.771 | 1449.578 |

### short_chat

| Backend | E2E tok/s median | Decode tok/s median | Latency ms median |
| --- | ---: | ---: | ---: |
| tinyllm | 393.798 | 473.240 | 1300.158 |
| transformers | 281.178 | 314.468 | 1820.911 |
| vllm | 705.089 | 757.570 | 726.150 |

## Relative goodput

For each window, `TTFT SLO = 2.0 × that window's 0.25C TTFT p99` and
`TPOT SLO = 1.5 × that window's 0.25C TPOT p99`. A request counts as good only if both
conditions hold. These are relative experiment thresholds, not production objectives.

## Reproduction

See `manifest.json` for the exact source revisions, file hashes, environment, configuration,
and complete preparation and execution commands. Real prompt text remains only in the ignored
server-side workload artifacts; `selection.json` contains source metadata and prompt hashes.

## Raw artifacts

The server-side archive contains prepared private workloads, capacity and replay workloads,
request events, and unsanitized run JSON. Its SHA-256 is recorded in `manifest.json`; the archive
is intentionally not committed to Git.
