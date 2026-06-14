#pragma once

#include <cstdint>
#include <vector>

#include "tiny_llm/runtime/processors.h"

namespace tiny_llm {

enum class RequestStatus {
    RUNNING = 0,
    WAITING = 1,
    FINISHED = 2,
};

struct Request {
    uint64_t request_id = 0;
    SamplingParams sampling_params;
    RequestStatus status = RequestStatus::WAITING;
    int32_t num_computed_tokens = 0;
    std::vector<int32_t> prompt_token_ids;
    std::vector<int32_t> context_token_ids;
};

} // namespace tiny_llm
