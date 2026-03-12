#include "core/tensor.h"

namespace tiny_llm {

size_t Tensor::numel() const {
    size_t n = 1;
    for (auto d : shape_) n *= d;
    return n;
}

} // namespace tiny_llm
