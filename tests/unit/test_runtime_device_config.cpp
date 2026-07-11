#include "tiny_llm/core/allocator.h"
#include "tiny_llm/core/context.h"
#include "tiny_llm/runtime/kv_cache.h"
#include "tiny_llm/runtime/parallel_config.h"
#include "tiny_llm/runtime/runtime_dtype.h"

#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>

TEST(RuntimeDeviceConfigTest, RuntimeObjectsCarryCpuDeviceConfig)
{
    tiny_llm::StackAllocator workspace(1024, tiny_llm::ParallelConfig::cpu());
    EXPECT_TRUE(workspace.parallel_config().is_cpu());
    EXPECT_TRUE(workspace.device().is_cpu());

    auto pool = std::make_unique<unsigned char[]>(4 * 64);
    tiny_llm::BlockAllocator blocks(4, 64, pool.get(), tiny_llm::ParallelConfig::cpu());
    EXPECT_TRUE(blocks.parallel_config().is_cpu());
    EXPECT_TRUE(blocks.device().is_cpu());
    EXPECT_NE(blocks.get_block_ptr(0), nullptr);

    tiny_llm::KVCache::Config kv_cfg;
    kv_cfg.num_layers = 2;
    kv_cfg.block_size_tokens = 16;
    tiny_llm::KVCache kv(kv_cfg, &blocks);
    EXPECT_TRUE(kv.parallel_config().is_cpu());
    EXPECT_TRUE(kv.device().is_cpu());

    tiny_llm::ExecutionContext ctx(nullptr, &workspace, &kv, tiny_llm::ParallelConfig::cpu());
    EXPECT_TRUE(ctx.parallel_config().is_cpu());
    EXPECT_TRUE(ctx.device().is_cpu());
}

TEST(RuntimeDeviceConfigTest, ExecutionContextUsesExplicitDeviceConfig)
{
    tiny_llm::StackAllocator workspace(1024, tiny_llm::ParallelConfig::cpu());
    tiny_llm::ExecutionContext ctx(nullptr, &workspace, nullptr, tiny_llm::ParallelConfig::cpu());
    EXPECT_TRUE(ctx.device().is_cpu());
}

TEST(RuntimeDeviceConfigTest, RuntimeDTypeNamesSizesAndParsingAreStable)
{
    EXPECT_STREQ(tiny_llm::runtime_dtype_name(tiny_llm::RuntimeDType::kFloat32), "float32");
    EXPECT_STREQ(tiny_llm::runtime_dtype_name(tiny_llm::RuntimeDType::kBFloat16), "bfloat16");
    EXPECT_EQ(tiny_llm::runtime_dtype_size(tiny_llm::RuntimeDType::kFloat32), sizeof(float));
    EXPECT_EQ(tiny_llm::runtime_dtype_size(tiny_llm::RuntimeDType::kBFloat16), sizeof(uint16_t));
    EXPECT_EQ(tiny_llm::parse_runtime_dtype("fp32"), tiny_llm::RuntimeDType::kFloat32);
    EXPECT_EQ(tiny_llm::parse_runtime_dtype("bf16"), tiny_llm::RuntimeDType::kBFloat16);
    EXPECT_THROW(tiny_llm::parse_runtime_dtype("float16"), std::runtime_error);
}
