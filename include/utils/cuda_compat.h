#pragma once

#if TINYLLM_ENABLE_CUDA
  #include <cuda_runtime.h>
#else
  // CPU-only build: provide minimal CUDA type stubs so headers compile.
  using cudaStream_t = void*;
#endif