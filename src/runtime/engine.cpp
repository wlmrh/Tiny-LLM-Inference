#include "tiny_llm/runtime/engine.h"

namespace tiny_llm {

LLMEngine::LLMEngine(Model* model,
                     ExecutionContext* ctx,
                     KVCache* kv,
                     Tokenizer* tokenizer,
                     int32_t max_generated_tokens,
                     SchedulerConfig scheduler_config)
    : core_(std::make_unique<EngineCore>(
          model,
          ctx,
          kv,
          tokenizer,
          max_generated_tokens,
          scheduler_config)),
      input_preprocessor_(
          core_->mutable_tokenizer_registry(),
          core_->model(),
          core_->default_max_generated_tokens()),
      output_preprocessor_(core_->tokenizer_registry())
{
}

uint64_t LLMEngine::add_request(const std::string& prompt,
                                const UserSamplingParams& user_params,
                                const std::string& ext_request_id)
{
    EngineCoreRequest request = input_preprocessor_.process_inputs(
        prompt,
        user_params,
        ext_request_id);

    core_->add_request(request);
    output_preprocessor_.add_request(request);
    return request.internal_id;
}

bool LLMEngine::has_unfinished_requests() const
{
    return output_preprocessor_.has_unfinished_requests();
}

std::vector<UserOutput> LLMEngine::step()
{
    const std::vector<EngineCoreOutput> core_outputs = core_->execute_step();
    std::vector<UserOutput> user_outputs = output_preprocessor_.process_outputs(core_outputs);

    for (const UserOutput& out : user_outputs)
    {
        if (out.is_finished)
        {
            core_->abort_request(out.internal_id, out.error_message);
        }
    }

    return user_outputs;
}

} // namespace tiny_llm
