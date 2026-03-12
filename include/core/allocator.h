#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tiny_llm {

class Tensor;
enum class DType;

// Monotonic workspace allocator backed by a contiguous GPU buffer.
// The allocator is reset once per inference step and does not support free().
class StackAllocator {
public:
    explicit StackAllocator(size_t pool_size);
    ~StackAllocator();

    // Resets the bump offset to zero; call at the beginning of each step.
    void reset();
    // Reserves a contiguous region in the workspace; returns nullptr on OOM.
    void* allocate(size_t bytes);

    // Creates a tensor whose storage is allocated from this workspace.
    Tensor make_tensor(std::vector<int64_t> shape, DType dtype);

private:
    // Base GPU pointer of the workspace pool.
    void* base_ptr_ = nullptr;
    // Total capacity in bytes.
    size_t total_size_ = 0;
    // Current bump offset in bytes.
    size_t offset_ = 0;
};

// Fixed-size block allocator used by paged KV cache.
class BlockAllocator {
public:
    // num_blocks: block count, block_size_bytes: size per block, gpu_pool: base GPU pointer.
    BlockAllocator(size_t num_blocks, size_t block_size_bytes, void* gpu_pool);

    // Returns a free block id, or -1 when no block is available.
    int32_t allocate_block();
    // Releases a previously allocated block id back to the free list.
    void free_block(int32_t block_id);
    // Returns the GPU pointer for a block id, or nullptr for invalid ids.
    void* get_block_ptr(int32_t block_id) const;

private:
    // Base GPU pointer for all blocks.
    void* gpu_pool_ = nullptr;
    // Number of blocks in the pool.
    size_t num_blocks_ = 0;
    // Size of each block in bytes.
    size_t block_size_ = 0;
    // LIFO free-list of block ids.
    std::vector<int32_t> free_list_;
};

} // namespace tiny_llm