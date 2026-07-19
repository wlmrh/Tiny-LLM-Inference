# Project Status

Tiny-LLM-Inference is a source-built, single-process offline inference runtime. It is intended for systems learning,
correctness work, and controlled performance experiments; it is not an HTTP service or a distributed serving stack.

## v0.1.0 release baseline

The v0.1.0 release candidate was validated on an NVIDIA GeForce RTX 4080 SUPER with pinned SmolLM2-135M and
Qwen2.5-1.5B-Instruct snapshots. CPU, CUDA, real-model, FP32 generation, BF16 smoke, and three-backend token-alignment
gates passed. The release benchmark includes offline TinyLLM/Transformers/vLLM results and five 200-request TinyLLM
open-loop workloads. See the [v0.1.0 benchmark report](../benchmark/reports/v0.1.0/README.md) for the tested commit,
environment, model file hashes, workloads, raw JSON, strengths, and limitations.

The host runtime reported `32760 MiB` for a device named NVIDIA GeForce RTX 4080 SUPER, while
[NVIDIA's retail reference specification](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080-family/)
lists 16 GB GDDR6X. This discrepancy may be caused by cloud-platform device presentation or nonstandard
provisioning, but the exact cause was not independently verified. Hardware claims in this project therefore
refer to the provider-exposed benchmark host rather than a standard retail card configuration.

## realistic-v1 benchmark baseline

The realistic-v1 benchmark extends the release evidence without changing runtime behavior. It combines three
non-overlapping 1,000-request BurstGPT timing/length windows with length-matched OASST1 prompt content, calibrates
an experiment-local `C_ref` for each window, and replays each trace at 0.25, 0.50, 0.75, and 0.90 of that reference.
All 12 replays completed 1,000/1,000 requests with complete request metrics and zero reported errors. An eight-prompt
EOS-aware cohort produced exact pairwise token-ID agreement across TinyLLM, Transformers, and vLLM.

The offline results are workload-sensitive: TinyLLM exceeded the tested Transformers baseline for short chat,
medium chat, and long decode, but reached only 0.356x its median E2E throughput for long prefill; vLLM led every
performance cohort. Across the three replay windows, median relative good-request ratio declined from 0.998 at
0.25C_ref to 0.716 at 0.90C_ref. These results identify long-prefill and higher-load tail latency as current limits;
they do not establish production capacity or serving parity. See the
[realistic-v1 benchmark report](../benchmark/reports/realistic-v1/README.md) for the bound commit, environment,
source revisions, model hashes, workload composition, full per-window results, and limitations.

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

Performance reports apply only to their recorded model, dtype, hardware, workload, and software environment. Results
against Transformers and vLLM are baseline comparisons, not claims of production parity with vLLM or TensorRT-LLM.

## BF16 maturity

BF16 is an experimental CUDA path in v0.1.0. It has correctness coverage for GEMM boundaries, paged KV prefill/decode,
and end-to-end deterministic generation. The current implementation retains FP32 master weights and FP32 boundaries,
so it can use more memory and run slower than FP32 for small models. Benchmark both modes on the target workload.
