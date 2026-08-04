#pragma once

#include <cstdint>
#include <vector>

#include "tiny_llm/runtime/processors.h"

namespace tiny_llm
{

enum class RequestStatus
{
    RUNNING = 0,
    WAITING = 1,
    FINISHED = 2,
};

struct Request
{
    // Internal identifier assigned when the request enters the engine.
    uint64_t request_id = 0;
    // Sampling configuration associated with this request.
    SamplingParams sampling_params;
    // Current request lifecycle state.
    RequestStatus status = RequestStatus::WAITING;
    int32_t num_computed_tokens = 0;
    std::vector<int32_t> prompt_token_ids;
    std::vector<int32_t> context_token_ids;
};

} // namespace tiny_llm
