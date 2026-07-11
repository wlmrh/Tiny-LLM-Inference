#pragma once

#include <cstdint>

#include "tiny_llm/core/context.h"
#include "tiny_llm/operators/paged_attention.h"
#include "tiny_llm/runtime/runtime_stats.h"

namespace tiny_llm
{

/**
 * @brief Explicit model runtime context for one ModelRunner invocation.
 *
 * This object keeps scheduler-prepared attention metadata outside the model
 * hierarchy while still making it available to attention blocks.
 */
class RuntimeContext
{
  public:
    RuntimeContext(ExecutionContext &execution, ops::PagedAttentionRuntimeMetadata attention_metadata,
                   RuntimeProfilingStats *profiling_stats = nullptr, bool profile_detail_enabled = false)
        : execution_(execution), attention_metadata_(attention_metadata), profiling_stats_(profiling_stats),
          profile_detail_enabled_(profile_detail_enabled)
    {
    }

    ExecutionContext &execution() const
    {
        return execution_;
    }
    c10::Device device() const
    {
        return execution_.device();
    }

    const ops::PagedAttentionRuntimeMetadata &attention_metadata() const
    {
        return attention_metadata_;
    }

    RuntimeProfilingStats *profiling_stats() const
    {
        return profiling_stats_;
    }
    bool profile_detail_enabled() const
    {
        return profile_detail_enabled_ && profiling_stats_ != nullptr;
    }

  private:
    ExecutionContext &execution_;
    ops::PagedAttentionRuntimeMetadata attention_metadata_{};
    RuntimeProfilingStats *profiling_stats_ = nullptr;
    bool profile_detail_enabled_ = false;
};

} // namespace tiny_llm
