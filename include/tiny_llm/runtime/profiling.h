#pragma once

#include <chrono>

#include <c10/core/Device.h>

#include "tiny_llm/runtime/runtime_stats.h"
#include "utils/cuda_compat.h"

namespace tiny_llm
{

class RuntimeContext;

class ScopedRuntimeProfile
{
  public:
    ScopedRuntimeProfile(RuntimeContext &ctx, double RuntimeProfilingStats::*field);
    ~ScopedRuntimeProfile();

    ScopedRuntimeProfile(const ScopedRuntimeProfile &) = delete;
    ScopedRuntimeProfile &operator=(const ScopedRuntimeProfile &) = delete;

  private:
    using Clock = std::chrono::steady_clock;

    RuntimeProfilingStats *stats_ = nullptr;
    double RuntimeProfilingStats::*field_ = nullptr;
    c10::Device device_{c10::kCPU};
    cudaStream_t stream_ = nullptr;
#if TINYLLM_ENABLE_CUDA
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t end_event_ = nullptr;
#endif
    bool enabled_ = false;
    Clock::time_point start_{};
};

} // namespace tiny_llm
