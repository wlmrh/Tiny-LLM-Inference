#include "tiny_llm/runtime/engine_core.h"

#include <stdexcept>
#include <utility>

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
          nullptr))
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

    const int32_t vocab_size = executor_->vocab_size();
    for (int32_t token_id : request.prompt_token_ids)
    {
        if (token_id < 0 || token_id >= vocab_size)
        {
            throw std::runtime_error("EngineCore::add_request: prompt token is out of model vocab range.");
        }
    }

    Request scheduler_request;
    scheduler_request.request_id = request.internal_id;
    scheduler_request.prompt_token_ids = request.prompt_token_ids;
    scheduler_request._all_token_ids = request.prompt_token_ids;
    scheduler_request.sampling_params = request.sampling_params;
    scheduler_request.status = RequestStatus::WAITING;
    scheduler_->add_request(std::move(scheduler_request));
}

std::tuple<std::unordered_map<int, EngineCoreOutput>, bool> EngineCore::step()
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

    std::map<int, EngineCoreOutput> scheduler_outputs = scheduler_->update_from_output(scheduler_output, model_output);
    std::unordered_map<int, EngineCoreOutput> engine_core_outputs;
    engine_core_outputs.reserve(scheduler_outputs.size());
    for (auto& item : scheduler_outputs)
    {
        engine_core_outputs[item.first] = std::move(item.second);
    }

    return {std::move(engine_core_outputs), scheduler_output.total_num_scheduled_tokens > 0};
}

} // namespace tiny_llm
