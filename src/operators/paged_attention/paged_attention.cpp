#include "tiny_llm/operators/paged_attention.h"

#include <stdexcept>

namespace tiny_llm {
namespace ops {

void attention_paged(const Tensor&, Tensor&, ExecutionContext&) {
    throw std::runtime_error("attention_paged is not implemented yet.");
}

} // namespace ops
} // namespace tiny_llm
