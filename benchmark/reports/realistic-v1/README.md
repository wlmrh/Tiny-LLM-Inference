# TinyLLM Realistic Workload Benchmark v1

## Executive summary

This benchmark combines BurstGPT arrival/length traces with length-matched OASST1 prompts
to exercise heterogeneous request lengths, burst timing, and per-request output limits without
changing the inference engine. It is a single-process, single-device experiment, not a production HTTP SLA.

This narrative is rendered from the committed sanitized JSON; the benchmark was not rerun and
the raw measurements were not edited for this documentation update.

- `3` experiment-local reference-capacity calibrations and `12` trace replays completed.
- All trace replays completed 1,000/1,000 requests with zero reported errors: `yes`.
- `8` EOS-aware correctness prompts matched exactly across all three backends: `yes`.
- TinyLLM median E2E throughput was `1.40x`, `1.20x`, and `1.23x` the tested Transformers baseline for short chat, medium chat, and long decode,
  but only `0.36x` for long prefill. vLLM led all four performance cohorts.
- Completion remained 100% at every replay load, but tail latency and relative goodput degraded materially
  above 0.50C_ref; completion alone must not be read as an SLO or stability claim.

## Scope and claim boundary

Results apply only to the recorded commit, model, GPU, dtype, and workloads. Relative goodput
uses experiment-local thresholds and is not a production SLA or a claim of serving parity.
The labels `0.25C` through `0.90C` are retained in JSON for compatibility; in this report, `C_ref`
means the completion rate of one simultaneous 1,000-request calibration for the same window.
It is not a production saturation capacity derived from a steady-state load scan.

## Environment

- Runtime candidate commit: `272b8f0428abe0642ff72a79b5d9141a04964c53`
- git_dirty: `False`
- GPU / driver: `NVIDIA GeForce RTX 4080 SUPER; driver 595.71.05; host-reported memory 32760 MiB`
- CUDA toolkit: `Build cuda_12.8.r12.8/compiler.35583870_0`
- Suite Python / PyTorch / Transformers: `3.12.3 / 2.8.0+cu128 / 4.55.2`
- vLLM environment Python / PyTorch / Transformers / vLLM: `3.12.3 / 2.11.0 / 5.14.1 / 0.25.1`
- Model: `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- TinyLLM compute / KV dtype: `FP32 / FP32`
- Sampling: greedy; correctness retains EOS; performance uses fixed output and ignores EOS
- Trace warmup: `32 requests`; offline performance warmup/repeat: `1 / 1` per shard
- GPU memory disclosure: `32760 MiB` is the literal value returned by the benchmark host's
  runtime query, not the standard specification of a retail RTX 4080 SUPER.
  [NVIDIA's reference specification](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080-family/) lists `16 GB GDDR6X`.
  The mismatch may reflect cloud-platform device presentation or nonstandard provisioning,
  but the exact cause was not independently verified. Treat both the device name and memory
  capacity as provider-exposed properties of this benchmark host.

## Sources and workload construction

- OASST1 is a content proxy; it is not the original BurstGPT request text.
- BurstGPT token counts come from GPT services and are matched approximately under the Qwen tokenizer.
- Arrival timestamps are linearly scaled to the workload-specific measured reference rate.
- Fixed output lengths model requested inference work and ignore natural EOS behavior.
- Network, HTTP, authentication, multi-GPU, and production reliability overheads are excluded.

## Dataset manifest

- BurstGPT revision: `7eb2c4f8350f8a6985272386f5c14af1f678b299`
- BurstGPT SHA-256: `3326259f9efb11845bc5ef85fa97e6f691050b0974621c91ef22acd566c43a40`
- OASST1 revision: `fdf72ae0827c1cda404aff25b6603abec9e3399b`
- OASST1 SHA-256: `2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7`
- Qwen revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Windows: `3` × `1000` requests

## Selected trace windows

The windows come from the usable trace at the configured 10%, 50%, and 90% positions after filtering.
ISL is measured after applying the Qwen chat template; OSL is the fixed per-request generation limit.

| Window | Trace position | API / conversation | ISL tokens min / median / max | OSL tokens min / median / max | Input / output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| window-01 | 10% | 992 / 8 | 30 / 764.0 / 3344 | 7 / 11.0 / 506 | 767,687 / 12,910 |
| window-02 | 50% | 987 / 13 | 117 / 321.0 / 3193 | 7 / 7.0 / 505 | 338,446 / 10,394 |
| window-03 | 90% | 979 / 21 | 30 / 336.5 / 3633 | 20 / 234.0 / 512 | 617,458 / 271,082 |

Window-03 is decode-heavy: its requested output-token total is more than twenty times either
of the other windows. Request/s and `C_ref` therefore must be compared within a window, not
treated as workload-independent engine capacity.

## Cross-window trace summary

Each cell reports `min / median / max` across the three windows.

| Load | Achieved req/s | TTFT p99 ms | TPOT p99 ms | E2E p99 ms | Good ratio | Max concurrency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25C_ref | 0.187 / 0.583 / 1.737 | 6130.000 / 18830.780 / 21966.703 | 56.856 / 421.463 / 665.000 | 21130.000 / 22874.800 / 26213.380 | 0.995 / 0.998 / 1.000 | 7 / 50 / 218 |
| 0.50C_ref | 0.372 / 1.159 / 3.242 | 6221.140 / 19645.290 / 45207.500 | 94.427 / 653.500 / 729.580 | 24028.260 / 35652.500 / 46430.479 | 0.963 / 0.981 / 1.000 | 15 / 65 / 437 |
| 0.75C_ref | 0.554 / 1.577 / 4.554 | 14819.990 / 54901.300 / 61996.220 | 116.942 / 654.000 / 707.038 | 51749.300 / 55878.116 / 66959.000 | 0.697 / 0.847 / 0.897 | 27 / 170 / 521 |
| 0.90C_ref | 0.660 / 1.748 / 5.263 | 38969.310 / 57359.100 / 94571.670 | 126.524 / 654.000 / 707.345 | 58298.284 / 70610.760 / 99427.260 | 0.551 / 0.716 / 0.827 | 47 / 253 / 545 |

Median relative good-request ratio fell from 0.998 at 0.25C_ref to 0.981, 0.847, and 0.716
as load increased. Median TTFT p99 rose from 18.831 s at 0.25C_ref to 54.901 s at 0.75C_ref.
The knee between 0.50C_ref and 0.75C_ref is evidence of queueing pressure for these traces,
not a universal capacity threshold.

## Per-window trace details

`C_ref = 1000 × 1000 / measured_generation_ms` for each simultaneous calibration.

| Window | C_ref req/s | Load | Target req/s | Completed | Req/s | TTFT p99 ms | TPOT p99 ms | E2E p99 ms | Max conc. | Good ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window-01 | 2.329551 | 0.25C_ref | 0.582388 | 1000/1000 | 0.582774 | 18830.780 | 665.000 | 21130.000 | 50 | 1.0000 |
| window-01 | 2.329551 | 0.50C_ref | 1.164776 | 1000/1000 | 1.159201 | 19645.290 | 729.580 | 24028.260 | 65 | 1.0000 |
| window-01 | 2.329551 | 0.75C_ref | 1.747163 | 1000/1000 | 1.577297 | 61996.220 | 707.038 | 66959.000 | 170 | 0.6970 |
| window-01 | 2.329551 | 0.90C_ref | 2.096596 | 1000/1000 | 1.748016 | 94571.670 | 707.345 | 99427.260 | 253 | 0.5510 |
| window-02 | 7.164920 | 0.25C_ref | 1.791230 | 1000/1000 | 1.737281 | 21966.703 | 421.463 | 22874.800 | 218 | 0.9950 |
| window-02 | 7.164920 | 0.50C_ref | 3.582460 | 1000/1000 | 3.241544 | 45207.500 | 653.500 | 46430.479 | 437 | 0.9630 |
| window-02 | 7.164920 | 0.75C_ref | 5.373690 | 1000/1000 | 4.554024 | 54901.300 | 654.000 | 55878.116 | 521 | 0.8470 |
| window-02 | 7.164920 | 0.90C_ref | 6.448428 | 1000/1000 | 5.263407 | 57359.100 | 654.000 | 58298.284 | 545 | 0.8270 |
| window-03 | 0.747389 | 0.25C_ref | 0.186847 | 1000/1000 | 0.186750 | 6130.000 | 56.856 | 26213.380 | 7 | 0.9980 |
| window-03 | 0.747389 | 0.50C_ref | 0.373694 | 1000/1000 | 0.372148 | 6221.140 | 94.427 | 35652.500 | 15 | 0.9810 |
| window-03 | 0.747389 | 0.75C_ref | 0.560541 | 1000/1000 | 0.553596 | 14819.990 | 116.942 | 51749.300 | 27 | 0.8970 |
| window-03 | 0.747389 | 0.90C_ref | 0.672650 | 1000/1000 | 0.659970 | 38969.310 | 126.524 | 70610.760 | 47 | 0.7160 |

The JSON report additionally contains p50/p95/p99 queue, TTFT, engine TTFT, TPOT, and E2E
for the overall population and every log-type, ISL, and OSL bucket, together with input/output/total
token throughput, wall-clock duration, and cross-window min/median/max summaries.

## Offline correctness and three-backend cohorts

Correctness used `8` mixed prompts with EOS enabled. TinyLLM, Transformers, and
vLLM token IDs matched pairwise with zero mismatches: `yes`.
Performance cohorts used fixed output lengths and ignored EOS.

| Cohort | ISL tokens | OSL | Shards × requests |
| --- | ---: | ---: | ---: |
| Correctness | 30-3344 | 32, EOS-aware | 1 × 8 |
| Short chat | 43-256 | 64 | 3 × 8 |
| Medium chat | 257-1021 | 128 | 3 × 8 |
| Long prefill | 1029-3344 | 64 | 3 × 4 |
| Long decode | 43-1021 | 256 | 3 × 8 |

The following values are medians across the three deterministic shards, not confidence intervals.

| Cohort | Requests | TinyLLM E2E tok/s | Transformers | vLLM | Tiny/Transformers | Tiny/vLLM | TinyLLM latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short chat | 24 | 393.798 | 281.178 | 705.089 | 1.401x | 0.559x | 1300.158 |
| Medium chat | 24 | 307.089 | 255.891 | 706.412 | 1.200x | 0.435x | 3334.536 |
| Long prefill | 12 | 35.185 | 98.758 | 345.995 | 0.356x | 0.102x | 7275.857 |
| Long decode | 24 | 331.814 | 270.578 | 720.053 | 1.226x | 0.461x | 6172.127 |

## Interpretation

- Against the tested Transformers path, TinyLLM was strongest on short chat (1.40x),
  medium chat (1.20x), and long decode (1.23x) median E2E throughput.
- Long prefill is the clearest weakness: TinyLLM reached only 0.36x the Transformers
  median E2E throughput and roughly one tenth of vLLM's throughput in that cohort.
- vLLM led every performance cohort, so these data support a workload-sensitive engineering baseline,
  not a parity claim.
- All replays completed, but higher-load goodput and tail-latency degradation show that completion rate
  alone is insufficient for judging interactive service quality.

## Relative goodput

For each window, `TTFT SLO = 2.0 × that window's 0.25C_ref TTFT p99` and
`TPOT SLO = 1.5 × that window's 0.25C_ref TPOT p99`. A request counts as good only if both
conditions hold. These are relative experiment thresholds, not production objectives.

## Limitations

- Only three deterministic trace windows and one GPU/model/dtype configuration were measured.
- OASST1 supplies content, while BurstGPT supplies timing and target lengths; they are not original paired requests.
- BurstGPT lengths come from another tokenizer and are only approximately matched under Qwen.
- Arrival gaps are linearly scaled, and fixed-output performance runs do not model natural EOS.
- `C_ref` is a simultaneous-batch completion rate, not a steady-state production capacity estimate.
- Offline medians summarize three fixed shards; they do not provide randomized-order variance or confidence intervals.
- Network, request parsing, HTTP/gRPC, authentication, multi-GPU, failure recovery, and production reliability are excluded.

## Reproduction

See `manifest.json` for the exact source revisions, file hashes, environment, configuration,
and complete preparation and execution commands. Real prompt text remains only in the ignored
server-side workload artifacts; `selection.json` contains source metadata and prompt hashes.

## Raw artifacts

The server-side archive contains prepared private workloads, capacity and replay workloads,
request events, and unsanitized run JSON. Its SHA-256 is recorded in `manifest.json`; the archive
is intentionally not committed to Git.

- Archive: `tinyllm-realistic-v1-artifacts-272b8f0.tar.gz`
- SHA-256: `3492e48f40961ed62f796123f5c0fbb8e25cf643e40a67d1e18da1c30fc7ccdb`
