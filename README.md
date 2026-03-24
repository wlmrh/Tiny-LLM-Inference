# Tiny-LLM-Inference

## Implemented (Stage 1)
- Tensor metadata view (non-owning)
- ExecutionContext step lifecycle
- StackAllocator / BlockAllocator
- KV paging metadata logic + tests

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

## What's next
- cuBLAS GEMM baseline for matmul path
- RMSNorm correctness + perf pass
- Micro-model one-step forward (prefill + 1 decode step)
- Basic CUDA launch-overhead and allocator micro-bench iteration