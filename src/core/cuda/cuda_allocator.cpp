#include "tiny_llm/core/allocator.h"

#include "utils/cuda_utils.h"

namespace tiny_llm {

StackAllocator::StackAllocator(size_t pool_size) : total_size_(pool_size) {
    CHECK_CUDA(cudaMalloc(&base_ptr_, pool_size));
}

StackAllocator::~StackAllocator() {
    if (base_ptr_ != nullptr) {
        CHECK_CUDA(cudaFree(base_ptr_));
    }
}

void* StackAllocator::allocate(size_t bytes) {
    if (offset_ + bytes > total_size_) {
        return nullptr;
    }
    void* ptr = static_cast<char*>(base_ptr_) + offset_;
    offset_ += bytes;
    return ptr;
}

void StackAllocator::reset() {
    offset_ = 0;
}

} // namespace tiny_llm
