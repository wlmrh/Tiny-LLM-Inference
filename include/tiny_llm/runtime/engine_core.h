#pragma once

#include <cstdint>
#include <map>
#include <memory>

#include "tiny_llm/runtime/executor.h"
#include "tiny_llm/runtime/processors.h"
#include "tiny_llm/runtime/scheduler.h"

namespace tiny_llm {

class ExecutionContext;
class KVCache;
class Model;
class Tokenizer;

/**
 * @brief Inner runtime core: scheduler, KV binding and model forward.
 */
class EngineCore {
public:
    EngineCore(Model* model,
               ExecutionContext* ctx,
               KVCache* kv,
               Tokenizer* tokenizer,
               int32_t default_max_generated_tokens,
               SchedulerConfig scheduler_config = SchedulerConfig{});
    ~EngineCore() = default;

    void add_request(const EngineCoreRequest& request);
    void abort_request(uint64_t internal_id);

    std::map<uint64_t, EngineCoreOutput> step();
    void post_step();

private:
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<ModelExecutor> executor_;
};

} // namespace tiny_llm
