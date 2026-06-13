#pragma once

#include <chrono>

#include "tiny_llm/runtime/runtime_context.h"

namespace tiny_llm {

class ScopedRuntimeProfile {
public:
    ScopedRuntimeProfile(RuntimeContext& ctx, double RuntimeProfilingStats::*field);
    ~ScopedRuntimeProfile();

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
