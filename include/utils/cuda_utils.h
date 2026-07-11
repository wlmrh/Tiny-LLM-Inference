#pragma once

#include "utils/cuda_compat.h"
#include <cstdio>
#include <cstdlib>

namespace tiny_llm
{
namespace utils
{

inline void check_cuda(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        std::printf("CUDA Error:\n");
        std::printf("    File:       %s\n", file);
        std::printf("    Line:       %d\n", line);
        std::printf("    Error code: %d\n", error_code);
        std::printf("    Error text: %s\n", cudaGetErrorString(error_code));
        std::exit(1);
    }
}

} // namespace utils
} // namespace tiny_llm

// Wraps a CUDA runtime call.
// Aborts the process with diagnostics on failure.
#define CHECK_CUDA(call) ::tiny_llm::utils::check_cuda((call), __FILE__, __LINE__)