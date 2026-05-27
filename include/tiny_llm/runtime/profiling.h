#pragma once

#include <chrono>

#include "tiny_llm/runtime/runtime_context.h"

#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace tiny_llm {

namespace detail {

inline void synchronize_device_for_runtime_profile(const c10::Device& device)
{
#if TINYLLM_ENABLE_CUDA
    if (device.is_cuda())
    {
        const int device_index = device.index() >= 0 ? static_cast<int>(device.index()) : 0;
        cudaSetDevice(device_index);
        cudaDeviceSynchronize();
    }
#else
    (void)device;
#endif
}

} // namespace detail

class ScopedRuntimeProfile {
public:
    ScopedRuntimeProfile(RuntimeContext& ctx, double RuntimeProfilingStats::*field)
        : ctx_(ctx), field_(field), enabled_(ctx.profile_detail_enabled())
    {
        if (enabled_)
        {
            detail::synchronize_device_for_runtime_profile(ctx_.device());
            start_ = Clock::now();
        }
    }

    ~ScopedRuntimeProfile()
    {
        if (!enabled_)
        {
            return;
        }
        detail::synchronize_device_for_runtime_profile(ctx_.device());
        const auto end = Clock::now();
        RuntimeProfilingStats* stats = ctx_.profiling_stats();
        if (stats != nullptr)
        {
            stats->*field_ += std::chrono::duration<double, std::milli>(end - start_).count();
        }
    }

    ScopedRuntimeProfile(const ScopedRuntimeProfile&) = delete;
    ScopedRuntimeProfile& operator=(const ScopedRuntimeProfile&) = delete;

private:
    using Clock = std::chrono::steady_clock;

    RuntimeContext& ctx_;
    double RuntimeProfilingStats::*field_;
    bool enabled_ = false;
    Clock::time_point start_{};
};

} // namespace tiny_llm
