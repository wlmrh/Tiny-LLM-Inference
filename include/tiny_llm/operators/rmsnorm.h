#pragma once

namespace tiny_llm {
class Tensor;
class ExecutionContext;

namespace ops {
void rmsnorm(const Tensor& x, const Tensor& w, Tensor& y, ExecutionContext& ctx);
}

} // namespace tiny_llm
