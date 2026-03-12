#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include "utils/cuda_compat.h"

namespace tiny_llm {

class BlockAllocator;

// Minimal paged KV cache metadata.
// Each sequence stores one page table per layer:
// logical block id -> physical block id.
class KVCache {
public:
    // num_layers: number of transformer layers.
    // block_size_tokens: token capacity per cache block.
    struct Config {
        int32_t num_layers = 0;
        int32_t block_size_tokens = 16;
    };

    KVCache(Config cfg, BlockAllocator* blocks)
        : cfg_(cfg), blocks_(blocks) {}

    void start_sequence(int32_t seq_id);
    void end_sequence(int32_t seq_id);

    // Ensures blocks exist up to token_pos and updates page tables as needed.
    // This API currently manages metadata only; K/V writes are handled elsewhere.
    void ensure_capacity(int32_t seq_id, int32_t layer_id, int32_t token_pos);

    // Returns the host-side page table for one sequence/layer pair.
    const std::vector<int32_t>& page_table(int32_t seq_id, int32_t layer_id) const;

private:
    struct SeqState {
        // page_tables[layer][logical_block] = physical_block_id
        std::vector<std::vector<int32_t>> page_tables;
    };

    // Static cache configuration.
    Config cfg_;
    // Non-owning allocator used to reserve physical blocks.
    BlockAllocator* blocks_ = nullptr;
    // Sequence id -> per-layer page tables.
    std::unordered_map<int32_t, SeqState> seqs_;
};

} // namespace tiny_llm