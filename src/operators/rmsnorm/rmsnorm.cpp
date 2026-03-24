#include "tiny_llm/operators/rmsnorm.h"

#include <stdexcept>

namespace tiny_llm {
namespace ops {

void rmsnorm(const Tensor&, const Tensor&, Tensor&, ExecutionContext&, float) {
#if TINYLLM_ENABLE_CUDA
    throw std::runtime_error("rmsnorm CUDA dispatch is not implemented yet.");
#else
    throw std::runtime_error("rmsnorm requires CUDA build (TINYLLM_ENABLE_CUDA=ON).");
#endif
}

} // namespace ops
} // namespace tiny_llm
