#pragma once

#include <cstddef>
#include <cstring>

#if TINYLLM_ENABLE_CUDA
  #include <cuda_runtime.h>
#else
  // CPU-only build: provide minimal CUDA type stubs so headers compile.
  // These are not functional—they exist only to allow headers to parse.

  using cudaStream_t = void*;

  enum cudaError_t {
      cudaSuccess = 0,
      cudaErrorInvalidValue = 1,
      cudaErrorMemoryAllocation = 2,
  };

  enum cudaMemcpyKind {
      cudaMemcpyHostToDevice = 1,
      cudaMemcpyDeviceToHost = 2,
      cudaMemcpyDeviceToDevice = 3,
  };

  // Stub functions that will cause link errors if actually called.
  inline cudaError_t cudaMalloc(void** devPtr, size_t size) {
      return cudaError_t::cudaSuccess;
  }

  inline cudaError_t cudaFree(void* devPtr) {
      return cudaError_t::cudaSuccess;
  }

  inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,
                                cudaMemcpyKind kind) {
      if (kind == cudaMemcpyHostToDevice ||
          kind == cudaMemcpyDeviceToHost) {
          std::memcpy(dst, src, count);
      }
      return cudaError_t::cudaSuccess;
  }

  inline cudaError_t cudaDeviceSynchronize() {
      return cudaError_t::cudaSuccess;
  }

  inline const char* cudaGetErrorString(cudaError_t error) {
      return "CUDA stub (CPU-only mode)";
  }
#endif