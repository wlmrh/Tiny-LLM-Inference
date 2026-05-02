#include "tiny_llm/core/allocator.h"

#include "tiny_llm/core/tensor.h"

#include <stdexcept>

namespace tiny_llm {

namespace {
// Define in an anonymous namespace to restrict visibility to this translation unit.
size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::kFloat16:
            return 2;
        case DType::kFloat32:
        case DType::kInt32:
            return 4;
    }
    return 0;
}

ParallelConfig default_allocator_parallel_config() {
    return ParallelConfig::cpu();
}

}

Tensor StackAllocator::make_tensor(std::vector<int64_t> shape, DType dtype) {
    size_t bytes = dtype_size(dtype);
    for (const int64_t dim : shape) {
        bytes *= static_cast<size_t>(dim);
    }

    void* ptr = allocate(bytes);
    if (ptr == nullptr) {
        throw std::runtime_error("StackAllocator: out of memory");
    }
    return make_tensor_from_blob(ptr, shape, dtype);
}

BlockAllocator::BlockAllocator(size_t num_blocks, size_t block_size_bytes, void* memory_pool)
    : BlockAllocator(num_blocks, block_size_bytes, memory_pool, default_allocator_parallel_config()) {
}

BlockAllocator::BlockAllocator(size_t num_blocks,
                               size_t block_size_bytes,
                               void* memory_pool,
                               ParallelConfig parallel_config)
    : memory_pool_(memory_pool),
      num_blocks_(num_blocks),
      block_size_(block_size_bytes),
      parallel_config_(parallel_config) {
    parallel_config_.validate();
    free_list_.reserve(num_blocks_);
    for (int32_t i = static_cast<int32_t>(num_blocks_) - 1; i >= 0; --i) {
        free_list_.push_back(i);
    }
}

int32_t BlockAllocator::allocate_block() {
    if (free_list_.empty()) {
        return -1;
    }
    const int32_t id = free_list_.back();
    free_list_.pop_back();
    return id;
}

void BlockAllocator::free_block(int32_t block_id) {
    if (block_id < 0 || block_id >= static_cast<int32_t>(num_blocks_)) {
        return;
    }
    free_list_.push_back(block_id);
}

void* BlockAllocator::get_block_ptr(int32_t block_id) const {
    if (block_id < 0 || block_id >= static_cast<int32_t>(num_blocks_)) {
        return nullptr;
    }
    return static_cast<char*>(memory_pool_) + (static_cast<size_t>(block_id) * block_size_);
}

size_t BlockAllocator::free_block_count() const {
    return free_list_.size();
}

size_t BlockAllocator::total_block_count() const {
    return num_blocks_;
}

} // namespace tiny_llm
