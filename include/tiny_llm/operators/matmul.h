#pragma once

#include "tiny_llm/core/tensor.h"

namespace tiny_llm {
class ExecutionContext;

namespace ops {
void gemm(const Tensor& a, const Tensor& b, Tensor& c, ExecutionContext& ctx);
}

} // namespace tiny_llm
