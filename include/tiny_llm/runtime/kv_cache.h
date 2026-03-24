#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace tiny_llm {

class BlockAllocator;

/**
 * @brief Metadata manager for paged KV cache.
 *
 * Inspired by vLLM-style page tables that map logical token blocks to physical blocks.
 *
 * @note CONTRACT:
 * - This class manages metadata mappings only; it does not copy raw K/V tensors.
 * - Physical blocks are owned by BlockAllocator and persist across steps.
 * - end_sequence() must be called to release per-sequence physical blocks.
 */
class KVCache {
public:
    /**
     * @brief Static configuration for KV cache metadata manager.
     */
    struct Config {
        /// Number of transformer layers.
        int32_t num_layers = 0;
        /// Token capacity of one physical KV block.
        int32_t block_size_tokens = 16;
    };

    /**
     * @brief Construct KV cache metadata manager.
     * @param cfg Static cache configuration.
     * @param blocks Non-owning block allocator handle.
     */
    KVCache(Config cfg, BlockAllocator* blocks) : cfg_(cfg), blocks_(blocks) {}

    /**
     * @brief Initialize metadata state for a new sequence id.
     */
    void start_sequence(int32_t seq_id);

    /**
     * @brief Release all physical blocks owned by a sequence id.
     */
    void end_sequence(int32_t seq_id);

    /**
     * @brief Ensure enough mapped blocks exist for token_pos in one layer.
     */
    void ensure_capacity(int32_t seq_id, int32_t layer_id, int32_t token_pos);

    /**
     * @brief Get host-side page table for a sequence/layer pair.
     */
    const std::vector<int32_t>& page_table(int32_t seq_id, int32_t layer_id) const;

private:
    /**
     * @brief Per-sequence metadata state.
     */
    struct SeqState {
        // page_tables[layer_id][logical_block_idx] = physical_block_id
        std::vector<std::vector<int32_t>> page_tables;
    };

    /// Static configuration.
    Config cfg_;
    /// Non-owning allocator for physical block reservation/release.
    BlockAllocator* blocks_ = nullptr;
    /// Active sequence states keyed by sequence id.
    std::unordered_map<int32_t, SeqState> seqs_;
};

} // namespace tiny_llm
