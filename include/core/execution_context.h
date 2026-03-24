#pragma once
#include "utils/cuda_compat.h"

namespace tiny_llm {

class StackAllocator;
class KVCache;

/**
 * @class ExecutionContext
 * @brief A runtime context shared across operators during a single decoding step.
 * * @note **Contract**: Workspace tensors are valid only within the current step. 
 * After calling @ref begin_step(), previous workspace allocations are considered 
 * invalid and may be overwritten.
 */
class ExecutionContext {
public:
    /**
     * @struct StepGuard
     * @brief RAII helper that automatically starts a new execution step upon construction.
     * * @note The current implementation does not perform specific actions on destruction 
     * (End-of-step actions).
     */
    struct StepGuard {
        ExecutionContext& ctx; ///< Reference to the bound context

        /**
         * @brief Constructs the guard and immediately calls ctx.begin_step().
         * @param c The execution context to manage.
         */
        explicit StepGuard(ExecutionContext& c) : ctx(c) { ctx.begin_step(); }
        ~StepGuard() = default;
    };

    /**
     * @brief Constructs an ExecutionContext.
     * @param stream The CUDA stream used for kernel launches.
     * @param ws Optional per-step workspace allocator (non-owning pointer).
     * @param kv Optional KV cache service (non-owning pointer).
     */
    ExecutionContext(cudaStream_t stream, StackAllocator* ws, KVCache* kv)
        : stream_(stream), ws_(ws), kv_(kv) {}

    /**
     * @brief Returns the CUDA stream bound to this context.
     * @return cudaStream_t The asynchronous execution stream.
     */
    cudaStream_t stream() const { return stream_; }

    /**
     * @brief Returns the workspace allocator.
     * @return StackAllocator* Pointer to the allocator, or nullptr if not set.
     */
    StackAllocator* workspace() const { return ws_; }

    /**
     * @brief Returns the KV cache handle.
     * @return KVCache* Pointer to the KV cache service, or nullptr if not set.
     */
    KVCache* kv() const { return kv_; }

    /**
     * @brief Starts a new step by resetting temporary workspace allocations.
     */
    void begin_step();

    /**
     * @brief Creates a guard that calls begin_step() immediately.
     * @return StepGuard An RAII object for step management.
     */
    StepGuard step_guard() { return StepGuard(*this); }

private:
    /// CUDA stream used for asynchronous execution.
    cudaStream_t stream_{0};
    
    /// Non-owning pointer to the workspace allocator.
    StackAllocator* ws_{nullptr};
    
    /// Non-owning pointer to the KV cache service.
    KVCache* kv_{nullptr};
};

} // namespace tiny_llm