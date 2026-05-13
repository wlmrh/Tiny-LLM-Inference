#include "tiny_llm/core/allocator.h"
#include "tiny_llm/core/context.h"
#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/execution_context.h"
#include "tiny_llm/runtime/kv_cache.h"
#include "tiny_llm/runtime/parallel_config.h"

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

    tiny_llm::EngineArgs args;
    args.workspace = &workspace;
    args.parallel_config = tiny_llm::ParallelConfig::cpu();
    tiny_llm::initialize_global_execution_context(args, &kv);
    tiny_llm::ExecutionContext& ctx =
        tiny_llm::require_global_execution_context("RuntimeDeviceConfigTest");
    EXPECT_TRUE(ctx.parallel_config().is_cpu());
    EXPECT_TRUE(ctx.device().is_cpu());
    tiny_llm::reset_global_execution_context();
}

TEST(RuntimeDeviceConfigTest, RejectsMismatchedWorkspaceDevice)
{
    tiny_llm::StackAllocator workspace(1024, tiny_llm::ParallelConfig::cpu());
    tiny_llm::EngineArgs args;
    args.workspace = &workspace;
    args.parallel_config = tiny_llm::ParallelConfig::cuda(0);
    EXPECT_THROW(tiny_llm::initialize_global_execution_context(args, nullptr), std::runtime_error);
    tiny_llm::reset_global_execution_context();
}
