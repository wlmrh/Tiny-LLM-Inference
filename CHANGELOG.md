# Changelog

All notable changes are documented here. Semantic versioning starts with the first release.

## [Unreleased]

- Continue performance work on fully BF16-resident model execution.

## [0.1.0] - Unreleased

- First supported source release of the single-process CPU/CUDA offline inference runtime.
- Reproducible CMake presets, CPU CI, Docker build, Apache-2.0 license, contribution and citation metadata.
- Explicit execution-context ownership and explicit paged-attention metadata without process/thread-local state.
- CPU and CUDA real-model alignment gates for logits, intermediate tensors, and deterministic generation.
- Experimental CUDA BF16 GEMM and BF16 paged KV cache paths with explicit dtype reporting.
- CUDA event-based detailed profiling and reproducible benchmark environment manifests.
