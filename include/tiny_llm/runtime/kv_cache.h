#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace tiny_llm {

class BlockAllocator;

class KVCache {
public:
    struct Config {
        int32_t num_layers = 0;
        int32_t block_size_tokens = 16;
    };

    KVCache(Config cfg, BlockAllocator* blocks) : cfg_(cfg), blocks_(blocks) {}

    void start_sequence(int32_t seq_id);
    void end_sequence(int32_t seq_id);
    void ensure_capacity(int32_t seq_id, int32_t layer_id, int32_t token_pos);
    const std::vector<int32_t>& page_table(int32_t seq_id, int32_t layer_id) const;

private:
    struct SeqState {
        std::vector<std::vector<int32_t>> page_tables;
    };

    Config cfg_;
    BlockAllocator* blocks_ = nullptr;
    std::unordered_map<int32_t, SeqState> seqs_;
};

} // namespace tiny_llm
