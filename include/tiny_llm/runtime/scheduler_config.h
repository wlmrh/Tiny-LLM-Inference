#pragma once

#include <cstddef>
#include <cstdint>

namespace tiny_llm {

/**
 * @brief Config-driven scheduler settings shared by runtime construction paths.
 */
struct SchedulerConfig {
    size_t max_running_requests = 0; // 0 means no explicit limit
    bool enable_preemption = true;
    int32_t max_prefill_tokens_per_step = 256;
};

} // namespace tiny_llm
