#include "tiny_llm/runtime/engine.h"

#include "tiny_llm/runtime/engine_core.h"

namespace tiny_llm {

LLMEngine::LLMEngine(const EngineArgs& args)
    : core_(std::make_unique<EngineCore>(args)),
      input_preprocessor_(args),
      output_preprocessor_(args)
{
}

LLMEngine::~LLMEngine() = default;
LLMEngine::LLMEngine(LLMEngine&&) noexcept = default;
LLMEngine& LLMEngine::operator=(LLMEngine&&) noexcept = default;

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

const RuntimeProfilingStats& LLMEngine::last_step_profile() const
{
    return core_->last_step_profile();
}

std::vector<UserOutput> LLMEngine::step()
{
    auto [core_outputs, has_scheduled_tokens] = core_->step();
    (void)has_scheduled_tokens;

    std::vector<UserOutput> user_outputs = output_preprocessor_.process_outputs(core_outputs);
    for (const UserOutput& output : user_outputs)
    {
        if (output.is_finished)
        {
            input_preprocessor_.release_request(output.external_id, output.internal_id);
        }
    }

    return user_outputs;
}

} // namespace tiny_llm
