#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "tiny_llm/runtime/runtime_stats.h"

namespace tiny_llm
{

/**
 * @brief Aggregated model execution results for one runtime step.
 */
struct ModelRunnerOutput
{
    std::unordered_map<uint64_t, int32_t> req_id_to_index;
    std::vector<int32_t> sampled_token_ids;
    RuntimeProfilingStats profiling;
};

} // namespace tiny_llm
