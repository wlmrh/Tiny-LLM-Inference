# Project Status

Tiny-LLM-Inference is a source-built, single-process offline inference runtime. It is intended for systems learning,
correctness work, and controlled performance experiments; it is not an HTTP service or a distributed serving stack.

## Supported

- Linux CPU and one selected NVIDIA CUDA device.
- LLaMA/SmolLM2-compatible and Qwen2-family decoder-only checkpoints.
- Hugging Face tokenizer JSON/SentencePiece and single or sharded safetensors.
- Chunked prefill/decode scheduling, paged KV cache, preemption, greedy and seeded sampling.

The public runtime dtype contract exposes independent `compute_dtype` and `kv_cache_dtype` options. Both default to
FP32. CUDA BF16 compute uses BF16 GEMMs with FP32 operator boundaries, while BF16 KV cache uses a dedicated paged
attention kernel. CPU BF16 requests fail explicitly and never silently fall back to FP32.

## Not yet supported

- Concurrent multi-engine use in one process.
- FP16 execution, fully BF16-resident weights/activations, or quantization.
- Tensor/pipeline parallelism, serving protocols, prefix caching, LoRA, or multimodal models.

Performance reports apply only to their recorded model, dtype, hardware, workload, and software environment. A result
against Transformers is a baseline comparison, not a claim of production parity with vLLM or TensorRT-LLM.

## BF16 maturity

BF16 is an experimental CUDA path in v0.1.0. It has correctness coverage for GEMM boundaries, paged KV prefill/decode,
and end-to-end deterministic generation. The current implementation retains FP32 master weights and FP32 boundaries,
so it can use more memory and run slower than FP32 for small models. Benchmark both modes on the target workload.
