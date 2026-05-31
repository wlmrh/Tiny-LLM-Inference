#include "tiny_llm/runtime/kv_cache.h"

#include "tiny_llm/core/allocator.h"

#include <stdexcept>

namespace tiny_llm {

KVCache::KVCache(Config cfg, BlockAllocator* blocks)
    : cfg_(cfg), blocks_(blocks)
{
    if (blocks_ != nullptr)
    {
        parallel_config_ = blocks_->parallel_config();
        parallel_config_.validate();
    }
}

KVCache::KVCache(Config cfg,
                 size_t num_blocks,
                 size_t block_size_bytes,
                 void* memory_pool)
    : cfg_(cfg),
      owned_blocks_(std::make_unique<BlockAllocator>(
          num_blocks,
          block_size_bytes,
          memory_pool)),
      blocks_(owned_blocks_.get()),
      parallel_config_(blocks_->parallel_config())
{
    parallel_config_.validate();
}

KVCache::KVCache(Config cfg,
                 size_t num_blocks,
                 size_t block_size_bytes,
                 void* memory_pool,
                 ParallelConfig parallel_config)
    : cfg_(cfg),
      owned_blocks_(std::make_unique<BlockAllocator>(
          num_blocks,
          block_size_bytes,
          memory_pool,
          parallel_config)),
      blocks_(owned_blocks_.get()),
      parallel_config_(parallel_config)
{
    parallel_config_.validate();
}

KVCache::~KVCache() = default;

size_t KVCache::free_block_count() const {
    if (blocks_ == nullptr) {
        return 0;
    }
    return blocks_->free_block_count();
}

size_t KVCache::total_block_count() const {
    if (blocks_ == nullptr) {
        return 0;
    }
    return blocks_->total_block_count();
}

size_t KVCache::block_size_bytes() const {
    if (blocks_ == nullptr) {
        return 0;
    }
    return blocks_->block_size_bytes();
}

void* KVCache::block_pool_base() const {
    if (blocks_ == nullptr) {
        return nullptr;
    }
    return blocks_->memory_pool();
}

void* KVCache::block_ptr(int32_t block_id) const {
    if (blocks_ == nullptr) {
        return nullptr;
    }
    return blocks_->get_block_ptr(block_id);
}

void KVCache::start_sequence(int32_t seq_id) {
    if (seqs_.find(seq_id) != seqs_.end()) {
        throw std::runtime_error("KVCache: sequence already exists.");
    }
    SeqState state;
    state.page_tables.resize(static_cast<size_t>(cfg_.num_layers));
    seqs_[seq_id] = std::move(state);
}

void KVCache::end_sequence(int32_t seq_id) {
    // Check if there is a seqence record with seq_id 
    auto it = seqs_.find(seq_id);
    if (it == seqs_.end()) {
        return;
    }

    // Iterate the pagetable to release all the physical_blocks occupied by seq_id
    for (auto& table : it->second.page_tables) {
        for (int32_t physical_block : table) {
            if (blocks_ != nullptr) {
                blocks_->free_block(physical_block);
            }
        }
    }
    seqs_.erase(it);
}

void KVCache::ensure_capacity(int32_t seq_id, int32_t layer_id, int32_t token_pos) {
    auto it = seqs_.find(seq_id);
    if (it == seqs_.end()) {
        throw std::runtime_error("KVCache: sequence not found.");
    }
    if (layer_id < 0 || layer_id >= cfg_.num_layers) {
        throw std::runtime_error("KVCache: layer id out of range.");
    }
    // get all the blocks allocated for layer_idth layer of the seq
    std::vector<int32_t>& table = it->second.page_tables[static_cast<size_t>(layer_id)];
    const int32_t required_blocks = (token_pos / cfg_.block_size_tokens) + 1;

    while (static_cast<int32_t>(table.size()) < required_blocks) {
        if (blocks_ == nullptr) {
            throw std::runtime_error("KVCache: block allocator is null.");
        }
        const int32_t id = blocks_->allocate_block();
        if (id < 0) {
            throw std::runtime_error("KVCache: out of cache blocks.");
        }
        table.push_back(id);
    }
}

const std::vector<int32_t>& KVCache::page_table(int32_t seq_id, int32_t layer_id) const {
    const auto it = seqs_.find(seq_id);
    if (it == seqs_.end()) {
        throw std::runtime_error("KVCache: sequence not found.");
    }
    if (layer_id < 0 || layer_id >= cfg_.num_layers) {
        throw std::runtime_error("KVCache: layer id out of range.");
    }
    return it->second.page_tables[static_cast<size_t>(layer_id)];
}

} // namespace tiny_llm
