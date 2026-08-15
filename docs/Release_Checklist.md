# v0.2.0 Release Checklist

## Source and metadata

- [x] Project version, changelog, citation metadata, and release badge are `0.2.0` dated 2026-08-15.
- [x] Apache-2.0 license, changelog, citation metadata, contribution guide, and project status are present.
- [x] Dependencies used through FetchContent are pinned and disconnected updates are supported.
- [x] Public headers use the `include/tiny_llm/` namespace layout.

## Release-candidate static validation

- [x] `git diff --check` and local-only `AGENTS.md` ignore rule.
- [x] CMake preset, Markdown link, JSON, SVG/XML, and architecture-generation checks.
- [x] v0.1.0 and realistic-v1 published report files match their manifest hashes.
- [x] Active tools, benchmark configs, and documentation contain no personal or provider-specific Python defaults and no legacy build commands.

## Required release validation

- [ ] Push the exact candidate commit and confirm public CPU CI succeeds.
- [ ] Confirm CPU model-independent and SmolLM2-135M model-backed tests on the candidate.
- [ ] Confirm the CUDA build, model-independent/model-backed tests, and FP32/BF16 generation smoke tests.
- [ ] Confirm Transformers logits, intermediate tensor, and deterministic generation alignment.
- [ ] Keep realistic-v1 performance claims bound to runtime candidate `c353eb4`; rerun before making any new candidate-HEAD performance claim.

## Release operator actions

- [ ] Review the candidate diff and merge only after all required validation gates pass.
- [ ] Create and verify signed tag `v0.2.0` on the final release commit.
- [ ] Publish GitHub Release notes from `CHANGELOG.md` and attach any new archived benchmark artifacts.

BF16 remains experimental in v0.2.0 because weights and most operator boundaries remain FP32. It is a correctness-
validated mixed-precision mode, not yet a universal performance win.
