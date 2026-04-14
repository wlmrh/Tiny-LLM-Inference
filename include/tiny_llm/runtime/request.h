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
    uint64_t priority = 0;
    SamplingParams sampling_params;
    RequestStatus status = RequestStatus::WAITING;
    int32_t num_computed = 0;
    std::vector<int32_t> prompt_token_ids;
    std::vector<int32_t> _all_token_ids;

    void reset_generated_tokens();
    int32_t generated_tokens() const;
    bool has_valid_token_layout() const;
};

} // namespace tiny_llm
