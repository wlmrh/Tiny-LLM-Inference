#include "tiny_llm/runtime/engine.h"

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

LLMEngine::LLMEngine(const EngineArgs& args)
    : core_(std::make_unique<EngineCore>(args)),
      input_preprocessor_(args.tokenizer, args.model, args.max_generated_tokens),
      output_preprocessor_(args.tokenizer)
{
}

LLMEngine::LLMEngine(Model* model,
                     ExecutionContext* ctx,
                     KVCache* kv,
                     Tokenizer* tokenizer,
                     int32_t max_generated_tokens,
                     SchedulerConfig scheduler_config)
    : LLMEngine(make_engine_args(
          model,
          ctx,
          kv,
          tokenizer,
          max_generated_tokens,
          scheduler_config))
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
    const auto core_outputs = core_->step();
    std::vector<UserOutput> user_outputs = output_preprocessor_.process_outputs(core_outputs);
    core_->post_step();

    return user_outputs;
}

} // namespace tiny_llm
