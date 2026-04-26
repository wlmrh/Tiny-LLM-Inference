#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_map>
#include "utils/cuda_compat.h"

namespace tiny_llm {

class BlockAllocator;

/**
 * @brief Metadata manager for Paged KV Cache.
 * * Inspired by vLLM, this service decouples logical token positions from physical GPU memory 
 * using a page table mapping mechanism.
 * * @note CONTRACT:
 * - This class manages METADATA (mappings) only; it does not perform raw memory copies.
 * - OWNERSHIP: Physical blocks are owned by the BlockAllocator and persist across steps.
 * - CLEANUP: end_sequence() must be explicitly called to prevent physical block leaks.
 */
class KVCache {
public:
    /**
     * @brief Static configuration for the KV Cache service.
     */
    struct Config {
        int32_t num_layers = 0;          ///< Total number of transformer layers in the model.
        int32_t block_size_tokens = 16;  ///< Number of tokens stored in a single physical block.
    };

    /**
     * @brief Construct a new KVCache object.
     * @param cfg The cache configuration settings.
     * @param blocks Pointer to the block allocator for physical memory management.
     */
    KVCache(Config cfg, BlockAllocator* blocks);

    /**
     * @brief Constructs KVCache and its internal BlockAllocator from pool parameters.
     */
    KVCache(Config cfg,
            size_t num_blocks,
            size_t block_size_bytes,
            void* gpu_pool);

    ~KVCache();

    /**
     * @brief Returns configured transformer layer count.
     */
    int32_t num_layers() const { return cfg_.num_layers; }

    /**
     * @brief Returns token capacity of each KV block.
     */
    int32_t block_size_tokens() const { return cfg_.block_size_tokens; }

    /**
     * @brief Returns currently free physical blocks in the shared pool.
     */
    size_t free_block_count() const;
    size_t block_size_bytes() const;
    void* block_ptr(int32_t block_id) const;

    /**
     * @brief Initializes metadata structures for a new request sequence.
     * @param seq_id Unique identifier for the sequence session.
     */
    void start_sequence(int32_t seq_id);

    /**
     * @brief Releases all physical blocks associated with a sequence back to the allocator.
     * @param seq_id Unique identifier for the sequence to be terminated.
     * @note This must be called at the end of every inference request to prevent memory leaks.
     */
    void end_sequence(int32_t seq_id);

    /**
     * @brief Dynamic capacity check: ensures physical blocks exist for the given token position.
     * * If the current logical position exceeds assigned blocks, new blocks are allocated 
     * from the BlockAllocator and the page table is updated.
     * * @param seq_id Unique identifier for the sequence.
     * @param layer_id The specific transformer layer index.
     * @param token_pos The logical index of the token being processed (0-indexed).
     */
    void ensure_capacity(int32_t seq_id, int32_t layer_id, int32_t token_pos);

    /**
     * @brief Retrieves the mapping table (page table) for a specific sequence and layer.
     * @param seq_id Unique identifier for the sequence.
     * @param layer_id The specific transformer layer index.
     * @return A constant reference to the page table for the `layer_id`th layer of `seq_id` sequence 
     */
    const std::vector<int32_t>& page_table(int32_t seq_id, int32_t layer_id) const;

private:
    /**
     * @brief Internal state for a single sequence, encapsulating all per-layer page tables.
     */
    struct SeqState {
        // page_tables[layer_id][logical_block_idx] = physical_block_id
        std::vector<std::vector<int32_t>> page_tables;
    };

    Config cfg_;                        ///< Cache configuration.
    std::unique_ptr<BlockAllocator> owned_blocks_; ///< Optional owned allocator when built from raw pool args.
    BlockAllocator* blocks_ = nullptr;  ///< Non-owning pointer to the physical block pool.
    std::unordered_map<int32_t, SeqState> seqs_; ///< Map of active sequence IDs to their states.
};

} // namespace tiny_llm
