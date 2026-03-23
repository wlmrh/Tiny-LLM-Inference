#include "tiny_llm/core/context.h"

#include "tiny_llm/core/allocator.h"

namespace tiny_llm {

void ExecutionContext::begin_step() {
    if (ws_) {
        ws_->reset();
    }
}

} // namespace tiny_llm
