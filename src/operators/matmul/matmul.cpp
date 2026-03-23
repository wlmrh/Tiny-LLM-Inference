#include "tiny_llm/operators/matmul.h"

#include <stdexcept>

namespace tiny_llm {
namespace ops {

void gemm(const Tensor&, const Tensor&, Tensor&, ExecutionContext&) {
#if TINYLLM_ENABLE_CUDA
    throw std::runtime_error("gemm CUDA dispatch is not implemented yet.");
#else
    throw std::runtime_error("gemm requires CUDA build (TINYLLM_ENABLE_CUDA=ON).");
#endif
}

} // namespace ops
} // namespace tiny_llm
