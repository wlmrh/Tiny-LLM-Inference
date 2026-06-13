#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/parallel_config.h"

namespace tiny_llm {

/**
 * @brief Monotonic bump-pointer workspace allocator backed by a contiguous buffer.
 * * This allocator manages short-lived memory for intermediate tensors during a single 
 * inference step. 
 * * @note CONTRACT:
 * - Tensors created here are NON-OWNING views; they do not free memory upon destruction.
 * - LIFECYCLE: Memory is valid ONLY until the next reset() or begin_step() call.
 * - No individual free() support to ensure O(1) allocation speed and zero fragmentation.
 */
class StackAllocator {
public:
    /**
     * @brief Construct a new StackAllocator with a fixed pool size.
     * @param pool_size Total capacity of the workspace in bytes.
     */
    explicit StackAllocator(size_t pool_size);
    StackAllocator(size_t pool_size, ParallelConfig parallel_config);
    
    /** @brief Destructor. Frees the underlying contiguous buffer. */
    ~StackAllocator();

    /**
     * @brief Resets the bump offset to zero. 
     * Must be called at the beginning of each inference step to recycle memory.
     */
    void reset();

    /**
     * @brief Reserves a contiguous region in the workspace pool.
     * @param bytes Requested allocation size in bytes.
     * @return void* A pointer to the allocated memory, or nullptr if pool capacity is exceeded.
     * @note Implementations should ensure returned pointers meet GPU alignment requirements (e.g., 256 bytes).
     */
    void* allocate(size_t bytes);

    /**
     * @brief Retrieves the maximum memory usage observed since construction.
     * @return size_t The peak offset in bytes (High-Water Mark).
     */
    #if TINYLLM_ENABLE_DEBUG_STATS
        size_t peak_bytes() const { return peak_offset_; }
    #endif

    /**
     * @brief Factory method to create a Tensor using memory from this workspace.
     * @param shape Dimensions of the tensor.
     * @param dtype Data type of the tensor elements.
     * @return Tensor A non-owning tensor view backed by workspace memory.
     */
    Tensor make_tensor(std::vector<int64_t> shape, DType dtype);

    /**
     * @brief Runtime device backing this workspace.
     */
    const ParallelConfig& parallel_config() const { return parallel_config_; }
    c10::Device device() const { return parallel_config_.torch_device(); }

private:
    void* base_ptr_ = nullptr;     ///< Base CPU/GPU pointer of the workspace pool.
    size_t total_size_ = 0;        ///< Total capacity in bytes.
    size_t offset_ = 0;            ///< Current bump offset in bytes.
    ParallelConfig parallel_config_{};
    #ifdef TINYLLM_ENABLE_DEBUG_STATS
        size_t peak_offset_ = 0;       ///< Maximum observed offset in bytes.
    #endif
};

/**
 * @brief Fixed-size block allocator for Paged KV Cache storage.
 * * Manages a pool of physical memory blocks that can be mapped to logical 
 * sequence positions.
 * * @note CONTRACT:
 * - KV blocks are PERSISTENT and outlive a single inference step.
 * - Management is handled by the KVCache service; blocks are released only when a sequence ends.
 */
class BlockAllocator {
public:
    /**
     * @brief Construct a new BlockAllocator using an externally provided memory pool.
     * @param num_blocks Total number of blocks to manage.
     * @param block_size_bytes Size of each individual block in bytes.
     * @param memory_pool Pointer to the pre-allocated CPU/GPU memory pool.
     * @param parallel_config Runtime device backing the memory pool.
     */
    BlockAllocator(size_t num_blocks,
                   size_t block_size_bytes,
                   void* memory_pool,
                   ParallelConfig parallel_config);

    /** * @brief Allocates a free physical block ID.
     * @return int32_t A valid block ID, or -1 if no blocks are available in the pool.
     */
    int32_t allocate_block();

    /**
     * @brief Releases a physical block ID back to the free list for future reuse.
     * @param block_id The ID of the block to be released.
     */
    void free_block(int32_t block_id);

    /**
     * @brief Translates a physical block ID into its actual GPU memory address.
     * @param block_id The ID of the physical block.
     * @return void* The GPU pointer to the block's start address, or nullptr if the ID is invalid.
     */
    void* get_block_ptr(int32_t block_id) const;

    /**
     * @brief Returns currently available block count.
     */
    size_t free_block_count() const;

    /**
     * @brief Returns total block capacity.
     */
    size_t total_block_count() const;

    /**
     * @brief Returns the base pointer of the contiguous block memory pool.
     */
    void* memory_pool() const { return memory_pool_; }

    /**
     * @brief Returns fixed byte size of each physical block.
     */
    size_t block_size_bytes() const { return block_size_; }

    /**
     * @brief Runtime device backing the physical block memory pool.
     */
    const ParallelConfig& parallel_config() const { return parallel_config_; }
    c10::Device device() const { return parallel_config_.torch_device(); }

private:
    void* memory_pool_ = nullptr;     ///< Base CPU/GPU pointer for the entire block pool.
    size_t num_blocks_ = 0;           ///< Total capacity in number of blocks.
    size_t block_size_ = 0;           ///< Fixed size of each block in bytes.
    ParallelConfig parallel_config_{};
    std::vector<int32_t> free_list_;  ///< LIFO list of currently available block IDs.
    std::vector<bool> allocated_;     ///< Per-block allocation state for double-free detection.
};

} // namespace tiny_llm
