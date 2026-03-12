#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>

namespace tiny_llm {

class BlockAllocator;

// 最小形态：每个序列每层有一个 page_table（logical->physical block id）
class KVCache {
public:
    struct Config {
        int32_t num_layers = 0;
        int32_t block_size_tokens = 16;
    };

    KVCache(Config cfg, BlockAllocator* blocks)
        : cfg_(cfg), blocks_(blocks) {}

    void start_sequence(int32_t seq_id);
    void end_sequence(int32_t seq_id);

    // 初版：先只做“确保 block 分配/页表更新”，K/V 写入可以后面再细化
    void ensure_capacity(int32_t seq_id, int32_t layer_id, int32_t token_pos);

    // attention kernel 需要：给出某 seq/layer 的页表（host 版）
    const std::vector<int32_t>& page_table(int32_t seq_id, int32_t layer_id) const;

private:
    struct SeqState {
        std::vector<std::vector<int32_t>> page_tables; // [layer][logical_block] = physical_id
    };

    Config cfg_;
    BlockAllocator* blocks_ = nullptr;
    std::unordered_map<int32_t, SeqState> seqs_;
};

} // namespace tiny_llm