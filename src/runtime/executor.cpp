#include "tiny_llm/runtime/executor.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/models/mini_llama.h"
#include "tiny_llm/models/model.h"
#include "tiny_llm/models/tiny_lm.h"
#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/execution_context.h"
#include "tiny_llm/runtime/kv_cache.h"
#include "tiny_llm/runtime/sampling.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace tiny_llm {

ModelExecutor::ModelExecutor(Model* model, ExecutionContext* ctx, KVCache* kv)
    : model_(model), kv_(kv)
{
    set_global_execution_context(ctx);
}

ModelExecutor::ModelExecutor(const EngineArgs& args, KVCache* kv)
    : kv_(kv)
{
    init_from_args(args);
}

ModelExecutor::~ModelExecutor()
{
    reset_global_execution_context();
}

void ModelExecutor::init_from_args(const EngineArgs& args)
{
    model_ = args.model;
    if (model_ == nullptr)
    {
        switch (args.model_type)
        {
            case EngineModelType::kTinyEmbeddingLM:
                if (args.tiny_lm_checkpoint_path.empty())
                {
                    throw std::runtime_error("ModelExecutor: tiny_lm_checkpoint_path must be provided when model_type is kTinyEmbeddingLM.");
                }
                owned_model_ = std::make_unique<TinyEmbeddingLM>(
                    TinyEmbeddingLM::from_checkpoint(args.tiny_lm_checkpoint_path));
                break;
            case EngineModelType::kMiniLLaMA:
                owned_model_ = std::make_unique<MiniLLaMA>(args.mini_llama_config);
                break;
            case EngineModelType::kPrebuilt:
            default:
                throw std::runtime_error("ModelExecutor: model pointer is null and no constructible model_type is configured.");
        }

        model_ = owned_model_.get();
    }

    initialize_global_execution_context(args, kv_);
}

void ModelExecutor::validate_handles() const
{
    if (model_ == nullptr || g_execution_context == nullptr)
    {
        throw std::runtime_error("ModelExecutor: model/context must be non-null.");
    }
}

int32_t ModelExecutor::vocab_size() const
{
    validate_handles();
    return model_->vocab_size();
}


void ModelExecutor::execute_model(const SchedulerOutput& scheduler_output)
{
    ModelRunnerOutput model_output;
    const size_t total_tasks =
        scheduler_output.scheduled_new_reqs.size()
        + scheduler_output.scheduled_cached_reqs.req_ids.size();
    model_output.req_ids.reserve(total_tasks);
    model_output.sampled_token_ids.reserve(total_tasks);
    model_output.req_id_to_index.reserve(total_tasks);

    // 对编号为 req_id 的请求,生成的结果为 sampled_token_id,将结果记录在 model_output 中
    auto append_result = [&](uint64_t req_id, int32_t sampled_token_id) {
        const int32_t index = static_cast<int32_t>(model_output.req_ids.size());
        model_output.req_ids.push_back(req_id);
        model_output.sampled_token_ids.push_back(sampled_token_id);
        model_output.req_id_to_index[req_id] = index;
    };

    for (const NewRequestData& new_req : scheduler_output.scheduled_new_reqs)
    {
        const auto count_it = scheduler_output.num_scheduled_tokens.find(new_req.req_id);
        if (count_it == scheduler_output.num_scheduled_tokens.end())
        {
            throw std::runtime_error("ModelExecutor::execute_model: missing token budget for scheduled new request.");
        }

        const int32_t scheduled_tokens = count_it->second;
        if (scheduled_tokens <= 0)
        {
            throw std::runtime_error("ModelExecutor::execute_model: scheduled token budget for new request must be positive.");
        }

        if (new_req.core_seq_id < 0)
        {
            throw std::runtime_error("ModelExecutor::execute_model: invalid core_seq_id for scheduled new request.");
        }

        const int32_t start_position = new_req.num_computed_tokens;
        const int32_t prompt_size = static_cast<int32_t>(new_req.prompt_token_ids.size());
        const int32_t end_position = std::min(prompt_size, start_position + scheduled_tokens);
        if (end_position <= start_position)
        {
            throw std::runtime_error("ModelExecutor::execute_model: invalid prefill window for scheduled new request.");
        }

        try
        {
            for (int32_t pos = start_position; pos < end_position; ++pos)
            {
                run_prefill_token(
                    new_req.core_seq_id,
                    new_req.prompt_token_ids[static_cast<size_t>(pos)],
                    pos);
            }
        }
        catch (const std::exception& ex)
        {
            throw std::runtime_error(
                "ModelExecutor::execute_model: prefill failed for req_id="
                + std::to_string(new_req.req_id) + ": " + ex.what());
        }

        append_result(new_req.req_id, -1);
    }

    const CachedRequestData& cached = scheduler_output.scheduled_cached_reqs;
    for (size_t i = 0; i < cached.req_ids.size(); ++i)
    {
        const uint64_t req_id = cached.req_ids[i];
        const auto count_it = scheduler_output.num_scheduled_tokens.find(req_id);
        if (count_it == scheduler_output.num_scheduled_tokens.end())
        {
            throw std::runtime_error("ModelExecutor::execute_model: missing token budget for scheduled cached request.");
        }

        const int32_t scheduled_tokens = count_it->second;
        if (scheduled_tokens <= 0)
        {
            throw std::runtime_error("ModelExecutor::execute_model: scheduled token budget for cached request must be positive.");
        }

        if (i >= cached.num_computed_tokens.size()
            || i >= cached.core_seq_ids.size()
            || i >= cached.input_token_ids.size())
        {
            throw std::runtime_error("ModelExecutor::execute_model: cached request metadata is incomplete.");
        }

        const int32_t core_seq_id = cached.core_seq_ids[i];
        int32_t input_token = cached.input_token_ids[i];
        int32_t sampled_token = input_token;
        const int32_t start_position = cached.num_computed_tokens[i];

        try
        {
            for (int32_t step_idx = 0; step_idx < scheduled_tokens; ++step_idx)
            {
                sampled_token = run_decode_and_sample(
                    core_seq_id,
                    input_token,
                    start_position + step_idx);
                input_token = sampled_token;
            }

            append_result(req_id, sampled_token);
        }
        catch (const std::exception& ex)
        {
            throw std::runtime_error(
                "ModelExecutor::execute_model: decode failed for req_id="
                + std::to_string(req_id) + ": " + ex.what());
        }
    }

    // Current runtime is synchronous; keep this staged buffer for API compatibility.
    staged_model_output_ = std::move(model_output);
}

ModelRunnerOutput ModelExecutor::sample_tokens(const SchedulerOutput& grammar_output)
{
    (void)grammar_output;

    if (!staged_model_output_.has_value())
    {
        throw std::runtime_error("ModelExecutor::sample_tokens: execute_model must be called first.");
    }

    ModelRunnerOutput output = std::move(*staged_model_output_);
    staged_model_output_.reset();
    return output;
}

void ModelExecutor::run_prefill_token(int32_t core_seq_id, int32_t token, int32_t position) const
{
    validate_handles();
    ExecutionContext& ctx = require_global_execution_context("ModelExecutor::run_prefill_token");

    if (token < 0 || token >= model_->vocab_size())
    {
        throw std::runtime_error("ModelExecutor::run_prefill_token: token is out of model vocab range.");
    }

    std::vector<int32_t> input_ids = {token};
    std::vector<int32_t> positions = {position};
    std::vector<float> logits(static_cast<size_t>(model_->vocab_size()), 0.0f);

    Tensor input_tensor(input_ids.data(), {1}, DType::kInt32);
    Tensor pos_tensor(positions.data(), {1}, DType::kInt32);
    Tensor logits_tensor(logits.data(), {1, model_->vocab_size()}, DType::kFloat32);

    if (kv_ != nullptr)
    {
        for (int32_t layer = 0; layer < kv_->num_layers(); ++layer)
        {
            kv_->ensure_capacity(core_seq_id, layer, position);
        }
    }

    auto guard = ctx.step_guard();
    (void)guard;
    model_->forward_step(input_tensor, pos_tensor, logits_tensor, ctx);
}

int32_t ModelExecutor::run_decode_and_sample(int32_t core_seq_id, int32_t token, int32_t position) const
{
    validate_handles();
    ExecutionContext& ctx = require_global_execution_context("ModelExecutor::run_decode_and_sample");

    if (token < 0 || token >= model_->vocab_size())
    {
        throw std::runtime_error("ModelExecutor::run_decode_and_sample: token is out of model vocab range.");
    }

    std::vector<int32_t> input_ids = {token};
    std::vector<int32_t> positions = {position};
    std::vector<float> logits(static_cast<size_t>(model_->vocab_size()), 0.0f);

    Tensor input_tensor(input_ids.data(), {1}, DType::kInt32);
    Tensor pos_tensor(positions.data(), {1}, DType::kInt32);
    Tensor logits_tensor(logits.data(), {1, model_->vocab_size()}, DType::kFloat32);

    if (kv_ != nullptr)
    {
        for (int32_t layer = 0; layer < kv_->num_layers(); ++layer)
        {
            kv_->ensure_capacity(core_seq_id, layer, position);
        }
    }

    {
        auto guard = ctx.step_guard();
        (void)guard;
        model_->forward_step(input_tensor, pos_tensor, logits_tensor, ctx);
    }

    const int32_t next_token = sample_argmax(logits.data(), model_->vocab_size());
    if (next_token < 0 || next_token >= model_->vocab_size())
    {
        throw std::runtime_error("ModelExecutor::run_decode_and_sample: sampled token is out of model vocab range.");
    }

    return next_token;
}

} // namespace tiny_llm
