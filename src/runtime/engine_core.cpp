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

    const SchedulerOutput scheduler_output = scheduler_->schedule();

    ModelOutput model_output;
    model_output.tasks.reserve(scheduler_output.tasks.size());

    for (const SchedulerTaskDescriptor& task : scheduler_output.tasks)
    {
        ModelTaskOutput task_output;
        task_output.internal_id = task.internal_id;
        task_output.is_prefill = task.is_prefill;

        try
        {
            if (task.is_prefill)
            {
                for (size_t i = 0; i < task.token_ids.size(); ++i)
                {
                    executor_->run_prefill_token(
                        task.core_seq_id,
                        task.token_ids[i],
                        task.start_position + static_cast<int32_t>(i));
                }
                task_output.processed_tokens = static_cast<int32_t>(task.token_ids.size());
            }
            else
            {
                if (task.token_ids.size() != 1)
                {
                    throw std::runtime_error("EngineCore::step: decode task must contain exactly one token.");
                }

                const int32_t sampled = executor_->run_decode_and_sample(
                    task.core_seq_id,
                    task.token_ids[0],
                    task.start_position);
                task_output.sampled_token_id = sampled;
                task_output.processed_tokens = 1;
            }
        }
        catch (const std::exception& ex)
        {
            task_output.has_error = true;
            task_output.error_message = ex.what();
        }

        model_output.tasks.push_back(std::move(task_output));
    }

    return scheduler_->update_from_output(scheduler_output, model_output);
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
