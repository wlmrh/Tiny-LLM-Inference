#include "core/allocator.h"
#include "utils/cuda_utils.h"

namespace tiny_llm {

StackAllocator::StackAllocator(size_t pool_size) : total_size_(pool_size) {
    CHECK_CUDA(cudaMalloc(&base_ptr_, pool_size));
    offset_ = 0;
}

StackAllocator::~StackAllocator() {
    if (base_ptr_) {
        CHECK_CUDA(cudaFree(base_ptr_));
    }
}

void* StackAllocator::allocate(size_t bytes) {
    if (offset_ + bytes > total_size_) {
        return nullptr; // out of memory
    }
    void* ptr = static_cast<char*>(base_ptr_) + offset_;
    offset_ += bytes;
    return ptr;
}

void StackAllocator::reset() {
    offset_ = 0;
}

Tensor StackAllocator::create_tensor(std::vector<int64_t> shape, DType dtype) {
    // Calculate required bytes based on shape and dtype
    size_t bytes = 0;
    switch (dtype) {
        case DType::kFloat32: bytes = 4; break;
        case DType::kFloat16: bytes = 2; break;
        case DType::kInt8: bytes = 1; break;
    }
    for (auto d : shape) bytes *= d;
    
    void* ptr = allocate(bytes);
    if (ptr == nullptr) {
        throw std::runtime_error("StackAllocator: out of memory");
    }
    return Tensor(std::move(shape), dtype, ptr);
}

BlockAllocator::BlockAllocator(size_t num_blocks, size_t block_size_bytes, void* gpu_pool)
    : gpu_pool_(gpu_pool), block_size_(block_size_bytes), num_blocks_(num_blocks) {
    free_list_.reserve(num_blocks);
    for (int i = num_blocks - 1; i >= 0; --i) {
        free_list_.push_back(i);
    }
}

BlockAllocator::~BlockAllocator() {
    // gpu_pool memory is managed externally, not freed here
}

int32_t BlockAllocator::allocate_block() {
    if (free_list_.empty()) return -1;
    int32_t id = free_list_.back();
    free_list_.pop_back();
    return id;
}

void BlockAllocator::free_block(int32_t block_id) {
    free_list_.push_back(block_id);
}

void* BlockAllocator::get_block_ptr(int32_t block_id) const {
    if (block_id < 0 || block_id >= static_cast<int32_t>(num_blocks_)) {
        return nullptr;
    }
    return static_cast<char*>(gpu_pool_) + block_id * block_size_;
}

Tensor BlockAllocator::create_block_tensor(std::vector<int64_t> shape, DType dtype) {
    // For now, allocate one block per tensor
    // In reality, one block may contain multiple KV pairs
    int32_t block_id = allocate_block();
    if (block_id < 0) {
        throw std::runtime_error("BlockAllocator: no free blocks");
    }
    void* ptr = get_block_ptr(block_id);
    return Tensor(std::move(shape), dtype, ptr);
}

} // namespace tiny_llm
