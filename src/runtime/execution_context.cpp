#include "tiny_llm/runtime/execution_context.h"

#include "tiny_llm/core/allocator.h"
#include "tiny_llm/core/context.h"
#include "tiny_llm/runtime/engine_args.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace tiny_llm {

ExecutionContext* g_execution_context = nullptr;

namespace {

std::unique_ptr<ExecutionContext> g_owned_execution_context;
std::unique_ptr<StackAllocator> g_owned_workspace;

} // namespace

void set_global_execution_context(ExecutionContext* ctx)
{
    g_execution_context = ctx;
}

void initialize_global_execution_context(const EngineArgs& args, KVCache* kv)
{
    g_owned_execution_context.reset();
    g_owned_workspace.reset();
    args.parallel_config.validate();

    if (args.ctx != nullptr)
    {
        g_execution_context = args.ctx;
        return;
    }

    StackAllocator* workspace = args.workspace;
    if (workspace != nullptr && workspace->parallel_config() != args.parallel_config)
    {
        throw std::runtime_error(
            "initialize_global_execution_context: workspace device does not match EngineArgs parallel_config.");
    }
    if (workspace == nullptr && args.workspace_pool_size > 0)
    {
        g_owned_workspace = std::make_unique<StackAllocator>(
            args.workspace_pool_size,
            args.parallel_config);
        workspace = g_owned_workspace.get();
    }

    g_owned_execution_context = std::make_unique<ExecutionContext>(
        args.execution_stream,
        workspace,
        kv,
        args.parallel_config);
    g_execution_context = g_owned_execution_context.get();
}

ExecutionContext& require_global_execution_context(const char* caller)
{
    if (g_execution_context == nullptr)
    {
        throw std::runtime_error(std::string(caller) + ": execution context is not initialized.");
    }
    return *g_execution_context;
}

ExecutionContext& resolve_execution_context(ExecutionContext& fallback_ctx)
{
    if (g_execution_context != nullptr)
    {
        return *g_execution_context;
    }
    return fallback_ctx;
}

void reset_global_execution_context()
{
    g_execution_context = nullptr;
    g_owned_execution_context.reset();
    g_owned_workspace.reset();
}

} // namespace tiny_llm
