#pragma once

#include <cstdint>

#include "tiny_llm/core/context.h"
#include "tiny_llm/operators/paged_attention.h"

namespace tiny_llm {

/**
 * @brief Explicit model runtime context for one ModelRunner invocation.
 *
 * This object keeps scheduler-prepared attention metadata outside the model
 * hierarchy while still making it available to attention blocks.
 */
class RuntimeContext {
public:
    RuntimeContext(ExecutionContext& execution,
                   ops::PagedAttentionRuntimeMetadata attention_metadata)
        : execution_(execution), attention_metadata_(attention_metadata) {}

    ExecutionContext& execution() const { return execution_; }
    KVCache* kv() const { return execution_.kv(); }
    const ParallelConfig& parallel_config() const { return execution_.parallel_config(); }
    c10::Device device() const { return execution_.device(); }

    const ops::PagedAttentionRuntimeMetadata& attention_metadata() const
    {
        return attention_metadata_;
    }

private:
    ExecutionContext& execution_;
    ops::PagedAttentionRuntimeMetadata attention_metadata_{};
};

} // namespace tiny_llm
