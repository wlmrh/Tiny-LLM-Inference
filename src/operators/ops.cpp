#include "operators/ops.h"

#if !TINYLLM_ENABLE_CUDA
void gemm(...) { throw std::runtime_error("gemm requires CUDA"); }
#endif

namespace tiny_llm {

// Operator kernels and dispatch wiring will be added in subsequent iterations.

} // namespace tiny_llm
