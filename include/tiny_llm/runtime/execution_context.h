#pragma once

namespace tiny_llm {

class ExecutionContext;
class KVCache;
struct EngineArgs;

// Global execution context that can be accessed by runtime/model/operator code paths.
extern ExecutionContext* g_execution_context;

void set_global_execution_context(ExecutionContext* ctx);
void initialize_global_execution_context(const EngineArgs& args, KVCache* kv);
ExecutionContext& require_global_execution_context(const char* caller);
ExecutionContext& resolve_execution_context(ExecutionContext& fallback_ctx);
void reset_global_execution_context();

} // namespace tiny_llm
