# Tiny-LLM-Inference v0.1.0 Benchmark Report

> Release-candidate placeholder. Every result below will be replaced from the archived v0.1.0 JSON reports after the pinned-model CPU/CUDA and three-backend validation gates complete.

## Executive Summary

Pending release validation.

## System Scope and Claim Boundary

Tiny-LLM-Inference is a single-process, single-device offline decoder-only inference runtime. These measurements evaluate correctness and controlled offline/open-loop workloads; they do not represent an HTTP service, distributed deployment, or production SLA.

## Hardware and Software Environment

Pending the release manifest generated on the GPU host.

## Model, Dtype, and Sampling Configuration

- SmolLM2-135M revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- Qwen2.5-1.5B-Instruct revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Headline TinyLLM compute/KV dtype: FP32/FP32
- Sampling: greedy
- Correctness mode: EOS enabled
- Performance mode: fixed output with EOS ignored

## Correctness Methodology

Pending three-backend token-agreement results.

## Offline Workloads and Results

Pending `offline.json`.

## Open-loop Capacity Definition and Results

Capacity is recomputed from the release-candidate TinyLLM decode-heavy result:

```text
C = batch * 1000 / avg_total_latency_ms
```

Pending `open_loop.json`.

## Interpretation

Pending results. The final report will discuss both favorable and unfavorable workloads without selecting only headline wins.

## Limitations

- Single-process and single-device only.
- No HTTP/gRPC serving layer or production request router.
- Open-loop request arrivals exercise the incremental offline API, not a network server.
- BF16 retains FP32 master weights and operator boundaries and is not assumed to improve performance.
- TinyLLM allocator memory fields are diagnostic and are not directly comparable with total-process memory from another backend.

## Reproduction Commands

Pending the exact commands copied from the archived reports.

## Raw Artifact Manifest

Pending `manifest.json` and GitHub Release asset checksums.
