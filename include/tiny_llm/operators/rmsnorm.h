#pragma once

namespace tiny_llm {
class Tensor;
class ExecutionContext;

namespace ops {
/**
 * feat(ops): implement optimized RMSNorm kernel
 * * @param x Input tensor
 * @param w Scaling weight tensor
 * @param y Output tensor
 * @param ctx Execution context for resource management
 * @param eps Small constant for numerical stability
 */
void rmsnorm(const Tensor& x, const Tensor& w, Tensor& y, ExecutionContext& ctx, float eps);
}

} // namespace tiny_llm