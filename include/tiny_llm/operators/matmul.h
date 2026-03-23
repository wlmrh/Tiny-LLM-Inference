#pragma once

namespace tiny_llm {
class Tensor;
class ExecutionContext;

namespace ops {
void gemm(const Tensor& a, const Tensor& b, Tensor& c, ExecutionContext& ctx);
}

} // namespace tiny_llm
