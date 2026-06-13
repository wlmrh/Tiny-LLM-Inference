#include "tiny_llm/runtime/engine_core.h"

#include "tiny_llm/runtime/model_runner.h"
#include "tiny_llm/runtime/scheduler.h"

#include <stdexcept>
#include <utility>

namespace tiny_llm {

EngineCore::EngineCore(const EngineArgs& args)
    : scheduler_(std::make_unique<Scheduler>(args)),
      runner_(std::make_unique<ModelRunner>(
          args,
          scheduler_->kv_cache()))
{
    if (args.max_generated_tokens <= 0)
    {
        throw std::runtime_error("EngineCore: default max_generated_tokens must be positive.");
    }

    if (!scheduler_ || !runner_)
    {
        throw std::runtime_error("EngineCore: scheduler/executor must be initialized.");
    }
}

EngineCore::~EngineCore() = default;

void EngineCore::add_request(const EngineCoreRequest& request)
{
    if (!scheduler_ || !runner_)
    {
        throw std::runtime_error("EngineCore::add_request: scheduler/executor must be initialized.");
    }

    runner_->validate_token_ids(request.prompt_token_ids, "EngineCore::add_request");

    Request scheduler_request;
    scheduler_request.request_id = request.internal_id;
    scheduler_request.prompt_token_ids = request.prompt_token_ids;
    scheduler_request.sampling_params = request.sampling_params;
    scheduler_->add_request(std::move(scheduler_request));
}

std::tuple<std::unordered_map<int, EngineCoreOutput>, bool> EngineCore::step()
{
    if (!scheduler_ || !runner_)
    {
        throw std::runtime_error("EngineCore::step: scheduler/executor must be initialized.");
    }

    last_step_profile_ = RuntimeProfilingStats{};
    if (!scheduler_->has_unfinished_requests())
    {
        return std::make_tuple(std::unordered_map<int, EngineCoreOutput>{}, false);
    }

    const SchedulerOutput scheduler_output = scheduler_->schedule();
    ModelRunnerOutput model_output = runner_->run(scheduler_output);
    last_step_profile_ = model_output.profiling;

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
