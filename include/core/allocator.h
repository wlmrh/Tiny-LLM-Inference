#pragma once

#include <cstddef>
#include <vector>
#include "tensor.h"

namespace tiny_llm {

class StackAllocator {
public:
    explicit StackAllocator(size_t pool_size);
    ~StackAllocator();

    // allocate raw bytes from the pool
    void* allocate(size_t bytes);
    
    // reset for next token (deallocation is implicit)
    void reset();
    
    // factory method to create Tensor with memory from this pool
    Tensor create_tensor(std::vector<int64_t> shape, DType dtype);

private:
    void* base_ptr_ = nullptr;
    size_t offset_ = 0;
    size_t total_size_ = 0;
};

class BlockAllocator {
public:
    // Initialize with GPU memory dedicated to KV cache blocks
    BlockAllocator(size_t num_blocks, size_t block_size_bytes, void* gpu_pool);
    ~BlockAllocator();

    // allocate a block index
    int32_t allocate_block();
    
    // free a block index back to free list
    void free_block(int32_t block_id);
    
    // get physical GPU pointer for a given block
    void* get_block_ptr(int32_t block_id) const;
    
    // factory method to create Tensor for KV cache
    Tensor create_block_tensor(std::vector<int64_t> shape, DType dtype);

private:
    void* gpu_pool_ = nullptr;              // base pointer to GPU pool for blocks
    std::vector<int32_t> free_list_;        // free block indices
    size_t block_size_ = 0;                 // bytes per block
    size_t num_blocks_ = 0;
};

} // namespace tiny_llm
