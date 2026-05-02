#include "tiny_llm/core/allocator.h"
#include "tiny_llm/core/context.h"
#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/execution_context.h"
#include "tiny_llm/runtime/kv_cache.h"
#include "tiny_llm/runtime/parallel_config.h"

#include <cstdlib>
#include <stdexcept>

namespace {

void expect_true(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
void expect_throws(Fn fn, const char* message)
{
    try
    {
        fn();
    }
    catch (const std::runtime_error&)
    {
        return;
    }
    throw std::runtime_error(message);
}

} // namespace

int main()
{
    tiny_llm::StackAllocator workspace(1024, tiny_llm::ParallelConfig::cpu());
    expect_true(workspace.parallel_config().is_cpu(), "workspace must record CPU config.");
    expect_true(workspace.device().is_cpu(), "workspace device must be CPU.");

    tiny_llm::BlockAllocator blocks(4, 64, std::malloc(4 * 64), tiny_llm::ParallelConfig::cpu());
    expect_true(blocks.parallel_config().is_cpu(), "block allocator must record CPU config.");
    expect_true(blocks.device().is_cpu(), "block allocator device must be CPU.");
    void* block_pool = blocks.get_block_ptr(0);
    expect_true(block_pool != nullptr, "block pointer must be non-null.");

    tiny_llm::KVCache::Config kv_cfg;
    kv_cfg.num_layers = 2;
    kv_cfg.block_size_tokens = 16;
    tiny_llm::KVCache kv(kv_cfg, &blocks);
    expect_true(kv.parallel_config().is_cpu(), "KVCache must inherit block allocator config.");
    expect_true(kv.device().is_cpu(), "KVCache device must be CPU.");

    tiny_llm::EngineArgs args;
    args.workspace = &workspace;
    args.parallel_config = tiny_llm::ParallelConfig::cpu();
    tiny_llm::initialize_global_execution_context(args, &kv);
    tiny_llm::ExecutionContext& ctx =
        tiny_llm::require_global_execution_context("test_runtime_device_config");
    expect_true(ctx.parallel_config().is_cpu(), "ExecutionContext must record EngineArgs config.");
    expect_true(ctx.device().is_cpu(), "ExecutionContext device must be CPU.");
    tiny_llm::reset_global_execution_context();

    tiny_llm::EngineArgs mismatched_workspace_args;
    mismatched_workspace_args.workspace = &workspace;
    mismatched_workspace_args.parallel_config = tiny_llm::ParallelConfig::cuda(0);
    expect_throws(
        [&]() {
            tiny_llm::initialize_global_execution_context(mismatched_workspace_args, &kv);
        },
        "mismatched workspace and EngineArgs device must throw.");

    tiny_llm::reset_global_execution_context();
    std::free(block_pool);
    return 0;
}
