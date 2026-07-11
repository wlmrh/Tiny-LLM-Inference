# Project Status

Tiny-LLM-Inference is a source-built, single-process offline inference runtime. It is intended for systems learning,
correctness work, and controlled performance experiments; it is not an HTTP service or a distributed serving stack.

## Supported

- Linux CPU and one selected NVIDIA CUDA device.
- LLaMA/SmolLM2-compatible and Qwen2-family decoder-only checkpoints.
- Hugging Face tokenizer JSON/SentencePiece and single or sharded safetensors.
- Chunked prefill/decode scheduling, paged KV cache, preemption, greedy and seeded sampling.

## Not yet supported

- Concurrent multi-engine use in one process.
- FP16/BF16 execution or quantization.
- Tensor/pipeline parallelism, serving protocols, prefix caching, LoRA, or multimodal models.

Performance reports apply only to their recorded model, dtype, hardware, workload, and software environment. A result
against Transformers is a baseline comparison, not a claim of production parity with vLLM or TensorRT-LLM.
