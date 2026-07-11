# Contributing

Tiny-LLM-Inference accepts focused changes that preserve documented runtime contracts.

1. Configure with `cmake --preset cpu-debug` or `cmake --preset cuda-release`.
2. Build and run the matching test preset.
3. Format modified C++/CUDA files with the repository `.clang-format`.
4. Include tests for behavior changes and benchmark evidence for performance claims.

Model-backed tests use `TINYLLM_HF_TINY_LLAMA_DIR` and report a skip when a local checkpoint is unavailable. Do not
commit model weights, build products, transient benchmark events, or machine-specific paths.

Pull requests must describe correctness, performance, compatibility, and validation impact. Benchmark claims must
include model, dtype, device, input/output tokens, warmup, repeats, and a raw report artifact.
