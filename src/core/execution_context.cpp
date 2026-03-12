#include "core/execution_context.h"
#include "core/allocator.h"

namespace tiny_llm {

void ExecutionContext::begin_step() {
    if (ws_) {
        ws_->reset();
    }
}

} // namespace tiny_llm
