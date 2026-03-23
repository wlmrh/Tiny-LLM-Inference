#pragma once

namespace tiny_llm {
class Tensor;
class ExecutionContext;

namespace ops {
void attention_paged(const Tensor& q, Tensor& out, ExecutionContext& ctx);
}

} // namespace tiny_llm
