#include "tiny_llm/runtime/request.h"

#include <algorithm>

namespace tiny_llm {

void Request::reset_generated_tokens()
{
    _all_token_ids = prompt_token_ids;
    status = RequestStatus::WAITING;
    num_computed = 0;
}

int32_t Request::generated_tokens() const
{
    if (_all_token_ids.size() < prompt_token_ids.size())
    {
        return 0;
    }

    return static_cast<int32_t>(_all_token_ids.size() - prompt_token_ids.size());
}

bool Request::has_valid_token_layout() const
{
    if (_all_token_ids.size() < prompt_token_ids.size())
    {
        return false;
    }

    return std::equal(prompt_token_ids.begin(), prompt_token_ids.end(), _all_token_ids.begin());
}

} // namespace tiny_llm
