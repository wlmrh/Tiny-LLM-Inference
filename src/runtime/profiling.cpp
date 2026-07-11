#include "tiny_llm/runtime/profiling.h"

#include "tiny_llm/runtime/runtime_context.h"

#include <stdexcept>

#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace tiny_llm
{
ScopedRuntimeProfile::ScopedRuntimeProfile(RuntimeContext &ctx, double RuntimeProfilingStats::*field)
    : stats_(ctx.profiling_stats()), field_(field), device_(ctx.device()),
      stream_(ctx.execution().stream()),
      enabled_(ctx.profile_detail_enabled() && field_ != nullptr)
{
    if (enabled_)
    {
#if TINYLLM_ENABLE_CUDA
        if (device_.is_cuda())
        {
            if (cudaEventCreate(&start_event_) != cudaSuccess || cudaEventCreate(&end_event_) != cudaSuccess ||
                cudaEventRecord(start_event_, stream_) != cudaSuccess)
            {
                throw std::runtime_error("ScopedRuntimeProfile: failed to create or record CUDA events.");
            }
            return;
        }
#endif
        start_ = Clock::now();
    }
}

ScopedRuntimeProfile::~ScopedRuntimeProfile()
{
    if (!enabled_)
    {
        return;
    }
#if TINYLLM_ENABLE_CUDA
    if (device_.is_cuda())
    {
        float elapsed_ms = 0.0f;
        const bool ok = cudaEventRecord(end_event_, stream_) == cudaSuccess &&
                        cudaEventSynchronize(end_event_) == cudaSuccess &&
                        cudaEventElapsedTime(&elapsed_ms, start_event_, end_event_) == cudaSuccess;
        cudaEventDestroy(start_event_);
        cudaEventDestroy(end_event_);
        if (ok)
        {
            stats_->*field_ += static_cast<double>(elapsed_ms);
        }
        return;
    }
#endif
    const auto end = Clock::now();
    stats_->*field_ += std::chrono::duration<double, std::milli>(end - start_).count();
}

} // namespace tiny_llm
