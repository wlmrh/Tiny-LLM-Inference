#pragma once
#include <cuda_runtime.h>

namespace tiny_llm {

class StackAllocator;
class KVCache;

class ExecutionContext {
public:
    struct StepGuard {
        ExecutionContext& ctx;
        explicit StepGuard(ExecutionContext& c) : ctx(c) { ctx.begin_step(); }
        ~StepGuard() = default;
    };

    ExecutionContext(cudaStream_t stream, StackAllocator* ws, KVCache* kv)
        : stream_(stream), ws_(ws), kv_(kv) {}

    cudaStream_t stream() const { return stream_; }
    StackAllocator* workspace() const { return ws_; }
    KVCache* kv() const { return kv_; }

    void begin_step() {
        if (ws_) ws_->reset();
    }

    StepGuard step_guard() { return StepGuard(*this); }

private:
    cudaStream_t stream_{0};
    StackAllocator* ws_{nullptr};
    KVCache* kv_{nullptr};
};

} // namespace tiny_llm