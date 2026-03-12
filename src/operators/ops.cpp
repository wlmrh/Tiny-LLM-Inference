#include "operators/ops.h"

#include <stdexcept>

namespace tiny_llm {
namespace ops {

#if !TINYLLM_ENABLE_CUDA
void rmsnorm(const Tensor&, const Tensor&, Tensor&, ExecutionContext&) {
    throw std::runtime_error("rmsnorm requires CUDA build (TINYLLM_ENABLE_CUDA=ON).");
}
void gemm(const Tensor&, const Tensor&, Tensor&, ExecutionContext&) {
    throw std::runtime_error("gemm requires CUDA build (TINYLLM_ENABLE_CUDA=ON).");
}
#endif

} // namespace ops
} // namespace tiny_llm