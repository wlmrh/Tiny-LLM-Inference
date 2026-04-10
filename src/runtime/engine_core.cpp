#include "tiny_llm/runtime/engine_core.h"

#include <stdexcept>

namespace tiny_llm {

namespace {

EngineArgs make_engine_args(Model* model,
                            ExecutionContext* ctx,
                            KVCache* kv,
                            Tokenizer* tokenizer,
                            int32_t max_generated_tokens,
                            SchedulerConfig scheduler_config)
{
    EngineArgs args;
    args.model = model;
    args.ctx = ctx;
    args.kv = kv;
    args.tokenizer = tokenizer;
    args.max_generated_tokens = max_generated_tokens;
    args.scheduler_config = scheduler_config;
    return args;
}

} // namespace

EngineCore::EngineCore(const EngineArgs& args)
    : scheduler_(std::make_unique<Scheduler>(args)),
      executor_(std::make_unique<ModelExecutor>(
          args,
          scheduler_ ? scheduler_->kv_cache() : nullptr))
{
    (void)args.tokenizer;

    if (args.max_generated_tokens <= 0)
    {
        throw std::runtime_error("EngineCore: default max_generated_tokens must be positive.");
    }

    if (!scheduler_ || !executor_)
    {
        throw std::runtime_error("EngineCore: scheduler/executor must be initialized.");
    }
}

EngineCore::EngineCore(Model* model,
                       ExecutionContext* ctx,
                       KVCache* kv,
                       Tokenizer* tokenizer,
                       int32_t default_max_generated_tokens,
                       SchedulerConfig scheduler_config)
    : EngineCore(make_engine_args(
          model,
          ctx,
          kv,
          tokenizer,
          default_max_generated_tokens,
          scheduler_config))
{
}

void EngineCore::add_request(const EngineCoreRequest& request)
{
    if (!scheduler_ || !executor_)
    {
        throw std::runtime_error("EngineCore::add_request: scheduler/executor must be initialized.");
    }

    scheduler_->add_request(request, executor_->vocab_size());
}

std::tuple<std::unordered_map<int, EngineCoreOutputs>, bool> EngineCore::step()
{
    if (!scheduler_ || !executor_)
    {
        throw std::runtime_error("EngineCore::step: scheduler/executor must be initialized.");
    }

    if (!scheduler_->has_unfinished_requests())
    {
        return {{}, false};
    }

    const SchedulerOutput scheduler_output = scheduler_->schedule();
    executor_->execute_model(scheduler_output);
    ModelRunnerOutput model_output = executor_->sample_tokens(scheduler_output);

    std::unordered_map<int, EngineCoreOutputs> engine_core_outputs = scheduler_->update_from_output(scheduler_output, model_output);

    return {std::move(engine_core_outputs), scheduler_output.total_num_scheduled_tokens > 0};
}

} // namespace tiny_llm
