#include "tiny_llm/runtime/engine_core.h"

#include <stdexcept>

namespace tiny_llm {

EngineCore::EngineCore(Model* model,
                       ExecutionContext* ctx,
                       KVCache* kv,
                       Tokenizer* tokenizer,
                       int32_t default_max_generated_tokens,
                       SchedulerConfig scheduler_config)
    : scheduler_(std::make_unique<Scheduler>(kv, scheduler_config)),
      executor_(std::make_unique<ModelExecutor>(model, ctx, kv))
{
    (void)tokenizer;

    if (default_max_generated_tokens <= 0)
    {
        throw std::runtime_error("EngineCore: default max_generated_tokens must be positive.");
    }

    if (!scheduler_ || !executor_)
    {
        throw std::runtime_error("EngineCore: scheduler/executor must be initialized.");
    }
}

void EngineCore::add_request(const EngineCoreRequest& request)
{
    if (!scheduler_ || !executor_)
    {
        throw std::runtime_error("EngineCore::add_request: scheduler/executor must be initialized.");
    }

    scheduler_->add_request(request, executor_->vocab_size());
}

void EngineCore::abort_request(uint64_t internal_id)
{
    if (!scheduler_)
    {
        throw std::runtime_error("EngineCore::abort_request: scheduler must be initialized.");
    }

    scheduler_->abort_request(internal_id);
}

std::map<uint64_t, EngineCoreOutput> EngineCore::step()
{
    if (!scheduler_ || !executor_)
    {
        throw std::runtime_error("EngineCore::step: scheduler/executor must be initialized.");
    }

    return scheduler_->step(*executor_);
}

void EngineCore::post_step()
{
    if (!scheduler_)
    {
        throw std::runtime_error("EngineCore::post_step: scheduler must be initialized.");
    }

    scheduler_->post_step();
}

} // namespace tiny_llm
