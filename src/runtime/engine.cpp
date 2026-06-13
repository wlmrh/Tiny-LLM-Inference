#include "tiny_llm/runtime/engine.h"

#include "tiny_llm/runtime/engine_core.h"

#include <stdexcept>

namespace tiny_llm {

LLMEngine::LLMEngine(const EngineArgs& args)
    : core_(std::make_unique<EngineCore>(args)),
    input_preprocessor_(args),
    output_preprocessor_(args)
{
}

LLMEngine::~LLMEngine() = default;

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
    auto [core_outputs, has_scheduled_tokens] = core_->step();
    last_step_profile_ = core_->last_step_profile();

    std::vector<UserOutput> user_outputs = output_preprocessor_.process_outputs(core_outputs);
    for (const UserOutput& output : user_outputs)
    {
        if (output.is_finished)
        {
            input_preprocessor_.release_request(output.external_id, output.internal_id);
        }
    }

    // Trigger one final core step to reclaim finished Scheduler/KV state.
    if (!output_preprocessor_.has_unfinished_requests())
    {
        auto [cleanup_outputs, cleanup_has_scheduled_tokens] = core_->step();
        if (!cleanup_outputs.empty() || cleanup_has_scheduled_tokens)
        {
            throw std::runtime_error("LLMEngine::step: unexpected outputs during cleanup step.");
        }
    }

    return user_outputs;
}

} // namespace tiny_llm
