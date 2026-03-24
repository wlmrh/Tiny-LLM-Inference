#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tiny_llm {

class Tensor;
enum class DType;

/**
 * @brief Monotonic bump-pointer workspace allocator backed by a contiguous buffer.
 *
 * This allocator manages short-lived memory for intermediate tensors during a single
 * inference step.
 *
 * @note CONTRACT:
 * - Tensors created here are non-owning views.
 * - Memory is valid only until the next reset() or begin_step() call.
 * - Individual free() is intentionally unsupported for O(1) allocation.
 */
class StackAllocator {
public:
    /**
     * @brief Construct a fixed-size workspace allocator.
     * @param pool_size Total workspace capacity in bytes.
     */
    explicit StackAllocator(size_t pool_size);

    /**
     * @brief Destructor. Releases underlying pool memory.
     */
    ~StackAllocator();

    /**
     * @brief Reset allocation offset to zero for next step reuse.
     */
    void reset();

    /**
     * @brief Reserve a contiguous memory region from workspace.
     * @param bytes Requested allocation bytes.
     * @return Pointer to allocated region or nullptr on OOM.
     */
    void* allocate(size_t bytes);

    /**
     * @brief Peak workspace usage since allocator construction.
     * @return High-water mark in bytes.
     */
    size_t peak_bytes() const { return peak_offset_; }

    /**
     * @brief Create a tensor view backed by workspace storage.
     * @param shape Tensor dimensions.
     * @param dtype Tensor element type.
     */
    Tensor make_tensor(std::vector<int64_t> shape, DType dtype);

private:
    /// Base CPU/GPU pointer of workspace pool.
    void* base_ptr_ = nullptr;
    /// Total capacity in bytes.
    size_t total_size_ = 0;
    /// Current bump offset in bytes.
    size_t offset_ = 0;
    /// Maximum observed offset in bytes.
    size_t peak_offset_ = 0;
};

/**
 * @brief Fixed-size block allocator used by paged KV cache.
 *
 * @note CONTRACT:
 * - KV blocks outlive a single step.
 * - Lifetime is managed by KVCache/BlockAllocator and sequence cleanup.
 */
class BlockAllocator {
public:
    /**
     * @brief Construct block allocator over external pool.
     * @param num_blocks Number of fixed-size blocks.
     * @param block_size_bytes Size of each block.
     * @param gpu_pool Base pool pointer.
     */
    BlockAllocator(size_t num_blocks, size_t block_size_bytes, void* gpu_pool);

    /**
     * @brief Allocate one free block id.
     * @return Block id, or -1 if no free block remains.
     */
    int32_t allocate_block();

    /**
     * @brief Return block id to free list.
     */
    void free_block(int32_t block_id);

    /**
     * @brief Resolve block id to pool address.
     * @return Pointer to block start, or nullptr for invalid id.
     */
    void* get_block_ptr(int32_t block_id) const;

private:
    /// Base pointer for all blocks.
    void* gpu_pool_ = nullptr;
    /// Number of blocks in pool.
    size_t num_blocks_ = 0;
    /// Per-block byte size.
    size_t block_size_ = 0;
    /// LIFO free-list of available block ids.
    std::vector<int32_t> free_list_;
};

} // namespace tiny_llm
