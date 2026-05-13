#include "tiny_llm/runtime/scheduler.h"
#include "tiny_llm/runtime/kv_cache.h"

#include <gtest/gtest.h>
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
    manager.init_owned(kNumLayers, kBlockSizeTokens, 4, kBlockBytes, pool.data());

    EXPECT_EQ(manager.free_block_count(), 4u);
    EXPECT_TRUE(manager.allocate_slots(1, false, 0, 3));
    EXPECT_EQ(manager.free_block_count(), 0u);

    std::vector<std::vector<int32_t>> block_tables;
    manager.refresh_block_tables(1, true, block_tables);
    ASSERT_EQ(block_tables.size(), 2u);
    EXPECT_EQ(block_tables[0].size(), 2u);
    EXPECT_EQ(block_tables[1].size(), 2u);
    EXPECT_NE(block_tables[0][0], block_tables[1][0]);

    std::vector<int32_t> first_layer;
    manager.refresh_block_table(1, true, first_layer);
    EXPECT_EQ(first_layer, block_tables[0]);
}

TEST(KVCacheManagerTest, ReleasesBlocksAndAllowsReuse)
{
    std::vector<unsigned char> pool(4 * kBlockBytes);
    tiny_llm::KVCacheManager manager;
    manager.init_owned(kNumLayers, kBlockSizeTokens, 4, kBlockBytes, pool.data());

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
    manager.init_owned(kNumLayers, kBlockSizeTokens, 2, kBlockBytes, pool.data());

    EXPECT_FALSE(manager.allocate_slots(1, false, 0, 3));
    EXPECT_EQ(manager.free_block_count(), 2u);
    EXPECT_TRUE(manager.allocate_slots(1, false, 0, 1));
    EXPECT_EQ(manager.free_block_count(), 0u);
}

TEST(KVCacheManagerTest, EstimatesAdditionalBlocksForPrefillAndDecode)
{
    std::vector<unsigned char> pool(6 * kBlockBytes);
    tiny_llm::KVCacheManager manager;
    manager.init_owned(kNumLayers, kBlockSizeTokens, 6, kBlockBytes, pool.data());

    EXPECT_EQ(manager.estimate_prefill_new_blocks(1, false, 3, 0, 2), 2u);
    ASSERT_TRUE(manager.allocate_slots(1, false, 0, 2));
    EXPECT_EQ(manager.estimate_prefill_new_blocks(1, true, 3, 2, 1), 2u);
    EXPECT_EQ(manager.estimate_append_new_blocks(1, true, 2), 2u);
}
