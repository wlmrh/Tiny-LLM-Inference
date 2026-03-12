#pragma once
#include "utils/cuda_compat.h"

namespace tiny_llm {

class StackAllocator;
class KVCache;

// Runtime context shared across operators during one decoding step.
class ExecutionContext {
public:
    // RAII helper that starts a step on construction.
    // The current implementation does not perform end-of-step actions.
    struct StepGuard {
        ExecutionContext& ctx;
        explicit StepGuard(ExecutionContext& c) : ctx(c) { ctx.begin_step(); }
        ~StepGuard() = default;
    };

    // stream: CUDA stream used for kernel launches.
    // ws: optional per-step workspace allocator.
    // kv: optional KV cache service.
    ExecutionContext(cudaStream_t stream, StackAllocator* ws, KVCache* kv)
        : stream_(stream), ws_(ws), kv_(kv) {}

    // Returns the CUDA stream bound to this context.
    cudaStream_t stream() const { return stream_; }
    // Returns the optional workspace allocator.
    StackAllocator* workspace() const { return ws_; }
    // Returns the optional KV cache handle.
    KVCache* kv() const { return kv_; }

    // Starts a new step by resetting temporary workspace allocations.
    void begin_step() {
        if (ws_) ws_->reset();
    }

    // Creates a guard that calls begin_step() immediately.
    StepGuard step_guard() { return StepGuard(*this); }

private:
    // Stream used for asynchronous CUDA execution.
    cudaStream_t stream_{0};
    // Non-owning workspace allocator pointer.
    StackAllocator* ws_{nullptr};
    // Non-owning KV cache pointer.
    KVCache* kv_{nullptr};
};

} // namespace tiny_llm