#include "tiny_llm/runtime/scheduler.h"
#include "tiny_llm/runtime/kv_cache.h"
#include "tiny_llm/core/allocator.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {
constexpr int32_t kNumLayers = 2;
constexpr int32_t kBlockSizeTokens = 2;
constexpr size_t kBlockBytes = 64;
}

TEST(KVCacheManagerTest, AllocatesAndRefreshesPerLayerBlockTables)
{
    std::vector<unsigned char> pool(4 * kBlockBytes);
    tiny_llm::KVCacheManager manager;
    manager.init_owned(kNumLayers, kBlockSizeTokens, 4, kBlockBytes, pool.data(), tiny_llm::ParallelConfig::cpu());

    EXPECT_EQ(manager.free_block_count(), 4u);
    EXPECT_TRUE(manager.allocate_slots(1, false, 0, 3));
    EXPECT_EQ(manager.free_block_count(), 0u);

    std::vector<std::vector<int32_t>> block_tables;
    manager.refresh_block_tables(1, true, block_tables);
    ASSERT_EQ(block_tables.size(), 2u);
    EXPECT_EQ(block_tables[0].size(), 2u);
    EXPECT_EQ(block_tables[1].size(), 2u);
    EXPECT_NE(block_tables[0][0], block_tables[1][0]);
}

TEST(KVCacheManagerTest, ReleasesBlocksAndAllowsReuse)
{
    std::vector<unsigned char> pool(4 * kBlockBytes);
    tiny_llm::KVCacheManager manager;
    manager.init_owned(kNumLayers, kBlockSizeTokens, 4, kBlockBytes, pool.data(), tiny_llm::ParallelConfig::cpu());

    ASSERT_TRUE(manager.allocate_slots(1, false, 0, 3));
    EXPECT_EQ(manager.free_block_count(), 0u);
    manager.end_sequence(1);
    EXPECT_EQ(manager.free_block_count(), 4u);
    EXPECT_TRUE(manager.allocate_slots(2, false, 0, 1));
    EXPECT_EQ(manager.free_block_count(), 2u);
}

TEST(KVCacheManagerTest, ReportsCapacityFailureWithoutLeakingBlocks)
{
    std::vector<unsigned char> pool(2 * kBlockBytes);
    tiny_llm::KVCacheManager manager;
    manager.init_owned(kNumLayers, kBlockSizeTokens, 2, kBlockBytes, pool.data(), tiny_llm::ParallelConfig::cpu());

    EXPECT_FALSE(manager.allocate_slots(1, false, 0, 3));
    EXPECT_EQ(manager.free_block_count(), 2u);
    EXPECT_TRUE(manager.allocate_slots(1, false, 0, 1));
    EXPECT_EQ(manager.free_block_count(), 0u);
}

TEST(KVCacheManagerTest, EstimatesAdditionalBlocksForPrefillAndDecode)
{
    std::vector<unsigned char> pool(6 * kBlockBytes);
    tiny_llm::KVCacheManager manager;
    manager.init_owned(kNumLayers, kBlockSizeTokens, 6, kBlockBytes, pool.data(), tiny_llm::ParallelConfig::cpu());

    EXPECT_EQ(manager.estimate_prefill_new_blocks(1, false, 3, 0, 2), 2u);
    ASSERT_TRUE(manager.allocate_slots(1, false, 0, 2));
    EXPECT_EQ(manager.estimate_prefill_new_blocks(1, true, 3, 2, 1), 2u);
    EXPECT_EQ(manager.estimate_append_new_blocks(1, true, 2), 2u);
}

TEST(KVCacheManagerTest, RejectsDuplicateSequenceStart)
{
    std::vector<unsigned char> pool(2 * kBlockBytes);
    tiny_llm::BlockAllocator blocks(2, kBlockBytes, pool.data(), tiny_llm::ParallelConfig::cpu());
    tiny_llm::KVCache::Config cfg;
    cfg.num_layers = 1;
    cfg.block_size_tokens = kBlockSizeTokens;
    tiny_llm::KVCache kv(cfg, &blocks);

    kv.start_sequence(1);
    EXPECT_THROW(kv.start_sequence(1), std::runtime_error);
    kv.end_sequence(1);
    EXPECT_NO_THROW(kv.start_sequence(1));
}

TEST(KVCacheManagerTest, BlockAllocatorRejectsInvalidOrDuplicateFree)
{
    std::vector<unsigned char> pool(2 * kBlockBytes);
    tiny_llm::BlockAllocator blocks(2, kBlockBytes, pool.data(), tiny_llm::ParallelConfig::cpu());

    EXPECT_THROW(blocks.free_block(0), std::runtime_error);
    const int32_t block = blocks.allocate_block();
    ASSERT_GE(block, 0);
    EXPECT_EQ(blocks.free_block_count(), 1u);
    blocks.free_block(block);
    EXPECT_EQ(blocks.free_block_count(), 2u);
    EXPECT_THROW(blocks.free_block(block), std::runtime_error);
    EXPECT_THROW(blocks.free_block(-1), std::runtime_error);
}
