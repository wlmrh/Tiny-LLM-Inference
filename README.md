# Tiny-LLM-Inference

## Implemented (Stage 2 - Minimal Runtime Chain)
- Tensor metadata view (non-owning)
- ExecutionContext step lifecycle
- StackAllocator / BlockAllocator
- KV paging metadata logic + tests
- Runtime model/tokenizer abstraction interfaces (`Model`, `Tokenizer`)
- Minimal `MiniLLaMA::forward_step` path
- Checkpoint-loaded `TinyEmbeddingLM` path for small-model bring-up
- CUDA GEMM kernel + dispatch path
- CUDA paged attention kernel (copy baseline) + dispatch path
- Single-device runtime engine with prefill + decode scheduling loop
- WordPiece tokenizer loader (`vocab.txt`) + whitespace tokenizer fallback
- Argmax sampler with tokenizer/model contract validation

## Build
CPU-only:
```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=OFF
cmake --build build -j
```

CUDA:
```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=ON
cmake --build build -j
```

## Run tests
```bash
cd build && ctest --output-on-failure
```

`test_kernels` (CUDA build only) validates RMSNorm, GEMM, and paged attention against CPU references.
`test_ops_cpu` validates GEMM and paged attention in CPU mode.

## Run minimal runtime demo
```bash
./build/llama_inference
```

The demo initializes KV paging metadata, runs two sequences through a prefill+decode loop,
and prints generated token strings.

## Run tiny checkpoint-loaded model demo
```bash
./build/tiny_lm_inference
```

Or pass custom paths:
```bash
./build/tiny_lm_inference assets/tiny_lm/vocab.txt assets/tiny_lm/tiny_lm_checkpoint.txt "tiny llm inference"
```

## Current boundaries
- Matmul and paged attention have concrete CUDA kernels, but are baseline implementations (not yet tuned).
- Tokenizer/sampling are intentionally minimal for architecture bring-up.
- Runtime currently targets single-process, single-device scheduling.

## What's next
- cuBLAS GEMM dispatch and kernel specialization
- Real paged-attention kernel and KV value reads
- Decode policies (top-k / top-p / temperature)
- Request admission and QoS-oriented scheduling