#include "tiny_llm/runtime/processors.h"

#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/tokenizer.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tiny_llm {

InputPreprocessor::InputPreprocessor(const EngineArgs& args)
    : tokenizer_(args.tokenizer), default_max_tokens_(args.max_generated_tokens)
{
    if (tokenizer_ == nullptr)
    {
        throw std::runtime_error("InputPreprocessor: tokenizer must be non-null.");
    }
    if (default_max_tokens_ <= 0)
    {
        throw std::runtime_error("InputPreprocessor: default_max_tokens must be positive.");
    }

    validate_tokenizer_contract();
}

EngineCoreRequest InputPreprocessor::process_inputs(const std::string& prompt,
                                                    const UserSamplingParams& user_params) const
{
    EngineCoreRequest request;
    request.internal_id = assign_internal_id();

    request.prompt_token_ids = tokenize(prompt);
    request.sampling_params = normalize_sampling_params(user_params);

    validate_prompt_tokens(request.prompt_token_ids);
    validate_sampling_params(request.sampling_params);

    return request;
}

uint64_t InputPreprocessor::assign_internal_id() const
{
    if (next_internal_id_ == std::numeric_limits<uint64_t>::max())
    {
        throw std::runtime_error("InputPreprocessor::assign_internal_id: exhausted internal id space.");
    }

    const uint64_t assigned = next_internal_id_;
    ++next_internal_id_;
    return assigned;
}

void InputPreprocessor::validate_tokenizer_contract() const
{
    const Tokenizer* tokenizer = tokenizer_;

    const int32_t tokenizer_vocab = tokenizer->vocab_size();
    if (tokenizer_vocab <= 0)
    {
        throw std::runtime_error("InputPreprocessor::validate_tokenizer_contract: tokenizer vocab size must be positive.");
    }

    const int32_t bos_id = tokenizer->bos_id();
    const int32_t eos_id = tokenizer->eos_id();
    const int32_t unk_id = tokenizer->unk_id();
    if (!tokenizer->is_valid_token_id(bos_id)
        || !tokenizer->is_valid_token_id(eos_id)
        || (unk_id >= 0 && !tokenizer->is_valid_token_id(unk_id)))
    {
        throw std::runtime_error("InputPreprocessor::validate_tokenizer_contract: tokenizer special token id is out of range.");
    }
}

std::vector<int32_t> InputPreprocessor::tokenize(const std::string& text) const
{
    return tokenizer_->encode(text);
}

SamplingParams InputPreprocessor::normalize_sampling_params(const UserSamplingParams& user_params) const
{
    SamplingParams params;
    static_cast<SamplingParamsCommon&>(params) =
        static_cast<const SamplingParamsCommon&>(user_params);
    if (user_params.max_tokens < 0)
    {
        throw std::runtime_error("InputPreprocessor::normalize_sampling_params: max_tokens must be >= 0.");
    }
    params.max_tokens = user_params.max_tokens > 0 ? user_params.max_tokens : default_max_tokens_;

    const int32_t eos_token = tokenizer_->eos_id();
    if (!params.ignore_eos
        && std::find(params.stop_token_ids.begin(), params.stop_token_ids.end(), eos_token)
        == params.stop_token_ids.end())
    {
        params.stop_token_ids.push_back(eos_token);
    }

    return params;
}

void InputPreprocessor::validate_prompt_tokens(const std::vector<int32_t>& token_ids) const
{
    const Tokenizer* tokenizer = tokenizer_;

    if (token_ids.empty())
    {
        throw std::runtime_error("InputPreprocessor::validate_prompt_tokens: prompt token ids must be non-empty.");
    }

    for (int32_t token_id : token_ids)
    {
        if (!tokenizer->is_valid_token_id(token_id))
        {
            throw std::runtime_error("InputPreprocessor::validate_prompt_tokens: token id out of tokenizer range.");
        }
    }
}

void InputPreprocessor::validate_sampling_params(const SamplingParams& sampling_params) const
{
    const Tokenizer* tokenizer = tokenizer_;

    if (sampling_params.temperature < 0.0f)
    {
        throw std::runtime_error("InputPreprocessor::validate_sampling_params: temperature must be >= 0.");
    }
    if (!(sampling_params.top_p > 0.0f && sampling_params.top_p <= 1.0f))
    {
        throw std::runtime_error("InputPreprocessor::validate_sampling_params: top_p must be in (0, 1].");
    }
    if (sampling_params.top_k < 0)
    {
        throw std::runtime_error("InputPreprocessor::validate_sampling_params: top_k must be >= 0.");
    }
    if (sampling_params.repetition_penalty <= 0.0f)
    {
        throw std::runtime_error("InputPreprocessor::validate_sampling_params: repetition_penalty must be positive.");
    }
    if (sampling_params.max_tokens <= 0)
    {
        throw std::runtime_error("InputPreprocessor::validate_sampling_params: max_tokens must be positive.");
    }

    for (int32_t token_id : sampling_params.stop_token_ids)
    {
        if (!tokenizer->is_valid_token_id(token_id))
        {
            throw std::runtime_error("InputPreprocessor::validate_sampling_params: stop token id out of tokenizer range.");
        }
    }
}

OutPreprocessor::OutPreprocessor(const EngineArgs& args)
    : tokenizer_(args.tokenizer)
{
    if (tokenizer_ == nullptr)
    {
        throw std::runtime_error("OutPreprocessor: tokenizer must be non-null.");
    }
}

void OutPreprocessor::add_request(const EngineCoreRequest& request)
{
    const Tokenizer* tokenizer = tokenizer_;
    if (tokenizer == nullptr)
    {
        throw std::runtime_error("OutPreprocessor::add_request: tokenizer is not available.");
    }

    if (request.internal_id == 0)
    {
        throw std::runtime_error("OutPreprocessor::add_request: internal_id must be non-zero.");
    }
    if (states_.find(request.internal_id) != states_.end())
    {
        throw std::runtime_error("OutPreprocessor::add_request: duplicated internal_id.");
    }

    std::unique_ptr<RequestState> state = std::make_unique<RequestState>();
    state->internal_id = request.internal_id;
    state->sampling_params = request.sampling_params;
    state->prompt_token_ids = request.prompt_token_ids;
    state->generated_token_ids.clear();
    state->decoded_prefix_len = 0;
    state->is_finished = false;
    state->finish_reason.clear();
    state->cached_text = tokenizer->decode(state->prompt_token_ids);
    state->decoded_prefix_len = state->cached_text.size();

    states_[request.internal_id] = std::move(state);
}

std::vector<UserOutput> OutPreprocessor::process_outputs(const std::vector<EngineCoreOutput>& core_outputs)
{
    std::vector<UserOutput> user_outputs;
    user_outputs.reserve(core_outputs.size());
    std::vector<uint64_t> finished_ids;

    for (const EngineCoreOutput& core : core_outputs)
    {
        UserOutput out;
        out.internal_id = core.internal_id;

        auto it = states_.find(core.internal_id);
        if (it == states_.end())
        {
            out.error_message = "OutPreprocessor::process_outputs: request state not found.";
            out.is_finished = true;
            out.finish_reason = "error";
            user_outputs.push_back(std::move(out));
            continue;
        }

        RequestState& state = *(it->second);

        if (state.is_finished)
        {
            out.text = state.cached_text;
            out.generated_token_ids = state.generated_token_ids;
            out.is_finished = true;
            out.finish_reason = state.finish_reason;
            user_outputs.push_back(std::move(out));
            continue;
        }

        out.delta_text = incremental_decode(state, core.new_token_id);
        check_stop_criteria(state, core.new_token_id);

        out.text = state.cached_text;
        out.generated_token_ids = state.generated_token_ids;
        out.is_finished = state.is_finished;
        out.finish_reason = state.finish_reason;
        if (state.is_finished)
        {
            finished_ids.push_back(state.internal_id);
        }
        user_outputs.push_back(std::move(out));
    }

    for (uint64_t id : finished_ids)
    {
        states_.erase(id);
    }

    return user_outputs;
}

bool OutPreprocessor::has_unfinished_requests() const
{
    for (const auto& item : states_)
    {
        if (!item.second->is_finished)
        {
            return true;
        }
    }
    return false;
}

std::string OutPreprocessor::incremental_decode(RequestState& state, int32_t new_token_id)
{
    const Tokenizer* tokenizer = tokenizer_;
    if (tokenizer == nullptr)
    {
        throw std::runtime_error("OutPreprocessor::incremental_decode: tokenizer is not available.");
    }

    if (!tokenizer->is_valid_token_id(new_token_id))
    {
        throw std::runtime_error("OutPreprocessor::incremental_decode: sampled token id out of tokenizer range.");
    }

    state.generated_token_ids.push_back(new_token_id);

    std::vector<int32_t> merged_tokens;
    merged_tokens.reserve(state.prompt_token_ids.size() + state.generated_token_ids.size());
    merged_tokens.insert(merged_tokens.end(), state.prompt_token_ids.begin(), state.prompt_token_ids.end());
    merged_tokens.insert(merged_tokens.end(), state.generated_token_ids.begin(), state.generated_token_ids.end());

    const std::string decoded = tokenizer->decode(merged_tokens);
    if (state.decoded_prefix_len > decoded.size())
    {
        throw std::runtime_error("OutPreprocessor::incremental_decode: decoded prefix length is invalid.");
    }

    const std::string delta = decoded.substr(state.decoded_prefix_len);
    state.cached_text = decoded;
    state.decoded_prefix_len = decoded.size();
    return delta;
}

bool OutPreprocessor::check_stop_criteria(RequestState& state, int32_t latest_token)
{
    const Tokenizer* tokenizer = tokenizer_;
    if (tokenizer == nullptr)
    {
        throw std::runtime_error("OutPreprocessor::check_stop_criteria: tokenizer is not available.");
    }

    if (!state.sampling_params.ignore_eos && latest_token == tokenizer->eos_id())
    {
        state.is_finished = true;
        state.finish_reason = "eos";
        return true;
    }

    if (std::find(state.sampling_params.stop_token_ids.begin(),
                  state.sampling_params.stop_token_ids.end(),
                  latest_token)
        != state.sampling_params.stop_token_ids.end())
    {
        state.is_finished = true;
        state.finish_reason = "stop_token";
        return true;
    }

    if (state.generated_token_ids.size() >= static_cast<size_t>(state.sampling_params.max_tokens))
    {
        state.is_finished = true;
        state.finish_reason = "length";
        return true;
    }

    return false;
}

} // namespace tiny_llm
