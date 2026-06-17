#pragma once

#include <chrono>

#include <c10/core/Device.h>

#include "tiny_llm/runtime/runtime_stats.h"

namespace tiny_llm {

class RuntimeContext;

class ScopedRuntimeProfile {
public:
    ScopedRuntimeProfile(RuntimeContext& ctx, double RuntimeProfilingStats::*field);
    ~ScopedRuntimeProfile();

    ScopedRuntimeProfile(const ScopedRuntimeProfile&) = delete;
    ScopedRuntimeProfile& operator=(const ScopedRuntimeProfile&) = delete;

private:
    using Clock = std::chrono::steady_clock;

    RuntimeProfilingStats* stats_ = nullptr;
    double RuntimeProfilingStats::*field_ = nullptr;
    c10::Device device_{c10::kCPU};
    bool enabled_ = false;
    Clock::time_point start_{};
};

} // namespace tiny_llm
