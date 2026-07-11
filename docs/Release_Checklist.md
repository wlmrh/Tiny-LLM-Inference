# v0.1.0 Release Checklist

## Source and metadata

- [x] Project version is `0.1.0`.
- [x] Apache-2.0 license, changelog, citation metadata, contribution guide, and project status are present.
- [x] Dependencies used through FetchContent are pinned and disconnected updates are supported.
- [x] Public headers use the `include/tiny_llm/` namespace layout.

## Required validation

- [x] CPU warnings-as-errors build.
- [x] CPU model-independent and SmolLM2-135M real-model tests.
- [x] CUDA build on an NVIDIA Ada GPU.
- [x] CUDA model-independent and SmolLM2-135M real-model tests.
- [x] Transformers logits, intermediate tensor, and deterministic generation alignment.
- [x] FP32 and BF16 compute/KV generation smoke tests.
- [x] Benchmark output records compute dtype, KV dtype, workload parameters, source revision, and environment.

## Release operator actions

- [ ] Run the regression benchmark preset on the release commit and archive its JSON/Markdown report.
- [ ] Confirm the Git working tree is clean and public CI is green.
- [ ] Push the release branch, review the diff, merge it, then create signed tag `v0.1.0`.
- [ ] Publish release notes from `CHANGELOG.md` and attach the archived benchmark report.

BF16 remains experimental in v0.1.0 because weights and most operator boundaries remain FP32. It is a correctness-
validated mixed-precision mode, not yet a universal performance win.
