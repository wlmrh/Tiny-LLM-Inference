#pragma once

#include "tiny_llm/runtime/parallel_config.h"
#include "utils/cuda_compat.h"

namespace tiny_llm {

class StackAllocator;
class KVCache;

/**
 * @class ExecutionContext
 * @brief Runtime context shared across operators during one decoding step.
 *
 * @note CONTRACT:
 * - Workspace tensors are valid only within the current step.
 * - After begin_step(), previous workspace allocations are invalid.
 */
class ExecutionContext {
public:
    /**
     * @struct StepGuard
     * @brief RAII helper that starts a new step on construction.
     */
    struct StepGuard {
        ExecutionContext& ctx;

        /**
         * @brief Construct guard and call begin_step().
         */
        explicit StepGuard(ExecutionContext& c) : ctx(c) { ctx.begin_step(); }
        ~StepGuard() = default;
    };

    /**
     * @brief Construct execution context.
     * @param stream CUDA stream for async kernel launches.
     * @param ws Optional per-step workspace allocator (non-owning).
     * @param kv Optional KV cache handle (non-owning).
     */
    ExecutionContext(cudaStream_t stream,
                     StackAllocator* ws,
                     KVCache* kv,
                     ParallelConfig parallel_config = ParallelConfig::cpu())
        : stream_(stream), ws_(ws), kv_(kv), parallel_config_(parallel_config) {}

    /**
     * @brief Get bound CUDA stream.
     */
    cudaStream_t stream() const { return stream_; }

    /**
     * @brief Get workspace allocator handle.
     */
    StackAllocator* workspace() const { return ws_; }

    /**
     * @brief Get KV cache handle.
     */
    KVCache* kv() const { return kv_; }

    /**
     * @brief Get runtime device and future parallelism configuration.
     */
    const ParallelConfig& parallel_config() const { return parallel_config_; }

    /**
     * @brief Get the torch device selected for runtime tensors.
     */
    c10::Device device() const { return parallel_config_.torch_device(); }

    /**
     * @brief Start a new step and recycle temporary workspace allocations.
     */
    void begin_step();

    /**
     * @brief Create RAII step guard that immediately calls begin_step().
     */
    StepGuard step_guard() { return StepGuard(*this); }

private:
    /// Stream used for asynchronous CUDA execution.
    cudaStream_t stream_{0};
    /// Non-owning workspace allocator pointer.
    StackAllocator* ws_{nullptr};
    /// Non-owning KV cache pointer.
    KVCache* kv_{nullptr};
    /// Runtime device and v1 single-device parallel configuration.
    ParallelConfig parallel_config_{};
};

} // namespace tiny_llm
