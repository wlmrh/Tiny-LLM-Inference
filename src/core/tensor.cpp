#include "core/tensor.h"
#include "utils/cuda_utils.h"

#include <cassert>

namespace tiny_llm {

Tensor::Tensor(std::vector<int64_t> shape, DType dtype, void* gpu_ptr)
    : data_ptr_(gpu_ptr), shape_(std::move(shape)), dtype_(dtype) {
    // Tensor does not manage memory lifecycle.
    // gpu_ptr must be allocated by an external allocator (StackAllocator, BlockAllocator, etc.)
    assert(gpu_ptr != nullptr && "Tensor requires valid GPU pointer from allocator");
}

size_t Tensor::numel() const {
    size_t n = 1;
    for (auto d : shape_) n *= d;
    return n;
}

size_t Tensor::size_in_bytes() const {
    size_t elems = numel();
    size_t dtype_size = 0;
    switch (dtype_) {
        case DType::kFloat32: dtype_size = 4; break;
        case DType::kFloat16: dtype_size = 2; break;
        case DType::kInt8: dtype_size = 1; break;
    }
    return elems * dtype_size;
}

}
