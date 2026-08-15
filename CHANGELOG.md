# Changelog

All notable changes are documented here. Semantic versioning starts with the first release.

## [0.2.0] - 2026-08-15

- Add the realistic-v1 BurstGPT/OASST1 workload pipeline, three-backend offline cohorts, and twelve 1,000-request
  trace replays with request-level latency evidence.
- Accelerate sufficiently large full and chunked-prefill attention segments with SDPA, including cached-prefix
  gathering and offset-causal masking.
- Fix incremental UTF-8 output decoding across token boundaries.
- Keep tokenizer special-token state behind its implementation boundary and clarify empty model-directory errors.
- Refresh architecture, benchmark, status, release, and reproduction documentation against the current runtime.
- Remove unbuilt legacy tensor/RMSNorm example sources and the unvalidated Docker packaging path.

## [0.1.0] - 2026-07-18

- First supported source release of the single-process CPU/CUDA offline inference runtime.
- Reproducible CMake presets, CPU CI, Docker build, Apache-2.0 license, contribution and citation metadata.
- Explicit execution-context ownership and explicit paged-attention metadata without process/thread-local state.
- CPU and CUDA real-model alignment gates for logits, intermediate tensors, and deterministic generation.
- Experimental CUDA BF16 GEMM and BF16 paged KV cache paths with explicit dtype reporting.
- CUDA event-based detailed profiling and reproducible benchmark environment manifests.
- Pinned SmolLM2-135M and Qwen2.5-1.5B-Instruct release snapshots with file-level SHA-256 manifests.
- Reproducible three-backend offline correctness/performance and TinyLLM open-loop release reports.
