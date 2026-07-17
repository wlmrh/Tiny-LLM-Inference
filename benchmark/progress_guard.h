#pragma once

#include <cstdint>
#include <stdexcept>

namespace tiny_llm::benchmark
{

inline void require_step_progress(bool has_unfinished_requests, int64_t scheduled_tokens)
{
    if (has_unfinished_requests && scheduled_tokens == 0)
    {
        throw std::runtime_error("open-loop benchmark made no progress with unfinished requests.");
    }
}

} // namespace tiny_llm::benchmark
