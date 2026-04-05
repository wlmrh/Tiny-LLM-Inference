#include "tiny_llm/runtime/processors.h"

#include "tiny_llm/models/model.h"
#include "tiny_llm/runtime/engine_core.h"
#include "tiny_llm/runtime/tokenizer.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tiny_llm {

namespace {

std::string default_external_id(uint64_t internal_id)
{
    return std::string("req-") + std::to_string(internal_id);
}

} // namespace

InputPreprocessor::InputPreprocessor(Tokenizer* tokenizer,
                                     const Model* model,
                                     int32_t default_max_tokens)
    : tokenizer_(tokenizer), model_(model), default_max_tokens_(default_max_tokens)
{
    if (tokenizer_ == nullptr || model_ == nullptr)
    {
        throw std::runtime_error("InputPreprocessor: tokenizer/model must be non-null.");
    }
    if (default_max_tokens_ <= 0)
    {
        throw std::runtime_error("InputPreprocessor: default_max_tokens must be positive.");
    }

    validate_model_tokenizer_contract();
}

InputPreprocessor::InputPreprocessor(TokenizerRegistry* tokenizer_registry,
                                     const Model* model,
                                     int32_t default_max_tokens)
    : InputPreprocessor(
          tokenizer_registry != nullptr ? tokenizer_registry->mutable_tokenizer() : nullptr,
          model,
          default_max_tokens)
{
}

EngineCoreRequest InputPreprocessor::process_inputs(const std::string& prompt,
                                                    const UserSamplingParams& user_params,
                                                    const std::string& ext_request_id) const
{
    EngineCoreRequest request;
    request.internal_id = assign_internal_id();
    bind_external_id(request, ext_request_id);

    const std::string formatted_prompt = apply_chat_template(prompt);
    request.prompt_token_ids = tokenize(formatted_prompt);
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

void InputPreprocessor::bind_external_id(EngineCoreRequest& request, const std::string& ext_request_id) const
{
    request.external_id = ext_request_id.empty() ? default_external_id(request.internal_id) : ext_request_id;

    const auto [it, inserted] =
        external_to_internal_id_.emplace(request.external_id, request.internal_id);
    if (!inserted)
    {
        throw std::runtime_error("InputPreprocessor::bind_external_id: duplicated external_id.");
    }
}

void InputPreprocessor::validate_model_tokenizer_contract() const
{
    const Tokenizer* tokenizer = tokenizer_;

    if (model_->vocab_size() <= 0)
    {
        throw std::runtime_error("InputPreprocessor::validate_model_tokenizer_contract: model vocab size must be positive.");
    }

    const int32_t tokenizer_vocab = tokenizer->vocab_size();
    if (tokenizer_vocab <= 0)
    {
        throw std::runtime_error("InputPreprocessor::validate_model_tokenizer_contract: tokenizer vocab size must be positive.");
    }

    if (tokenizer->is_fixed_vocab() && tokenizer_vocab != model_->vocab_size())
    {
        throw std::runtime_error("InputPreprocessor::validate_model_tokenizer_contract: tokenizer vocab size must match model vocab size.");
    }

    const int32_t bos_id = tokenizer->bos_id();
    const int32_t eos_id = tokenizer->eos_id();
    const int32_t unk_id = tokenizer->unk_id();
    if (!tokenizer->is_valid_token_id(bos_id)
        || !tokenizer->is_valid_token_id(eos_id)
        || !tokenizer->is_valid_token_id(unk_id))
    {
        throw std::runtime_error("InputPreprocessor::validate_model_tokenizer_contract: tokenizer special token id is out of range.");
    }

    const int32_t expected_bos = model_->expected_bos_id();
    if (expected_bos >= 0 && expected_bos != bos_id)
    {
        throw std::runtime_error("InputPreprocessor::validate_model_tokenizer_contract: tokenizer bos_id mismatches model checkpoint.");
    }

    const int32_t expected_eos = model_->expected_eos_id();
    if (expected_eos >= 0 && expected_eos != eos_id)
    {
        throw std::runtime_error("InputPreprocessor::validate_model_tokenizer_contract: tokenizer eos_id mismatches model checkpoint.");
    }

    const int32_t expected_unk = model_->expected_unk_id();
    if (expected_unk >= 0 && expected_unk != unk_id)
    {
        throw std::runtime_error("InputPreprocessor::validate_model_tokenizer_contract: tokenizer unk_id mismatches model checkpoint.");
    }
}

std::vector<int32_t> InputPreprocessor::tokenize(const std::string& text) const
{
    return tokenizer_->encode(text);
}

std::string InputPreprocessor::apply_chat_template(const std::string& text) const
{
    // Keep no-op behavior for pure text mode and fixed prompt formatting.
    return text;
}

SamplingParams InputPreprocessor::normalize_sampling_params(const UserSamplingParams& user_params) const
{
    SamplingParams params;
    params.temperature = user_params.temperature;
    params.top_p = user_params.top_p;
    params.top_k = user_params.top_k;
    params.max_tokens = user_params.max_tokens > 0 ? user_params.max_tokens : default_max_tokens_;
    params.stop_token_ids = user_params.stop_token_ids;
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
        if (token_id < 0 || token_id >= model_->vocab_size())
        {
            throw std::runtime_error("InputPreprocessor::validate_prompt_tokens: token id out of model vocab range.");
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
        if (token_id < 0 || token_id >= model_->vocab_size())
        {
            throw std::runtime_error("InputPreprocessor::validate_sampling_params: stop token id out of model vocab range.");
        }
    }
}

OutPreprocessor::OutPreprocessor(const Tokenizer* tokenizer)
    : tokenizer_(tokenizer)
{
    if (tokenizer_ == nullptr)
    {
        throw std::runtime_error("OutPreprocessor: tokenizer must be non-null.");
    }
}

OutPreprocessor::OutPreprocessor(const TokenizerRegistry* tokenizer_registry)
    : OutPreprocessor(tokenizer_registry != nullptr ? tokenizer_registry->tokenizer() : nullptr)
{
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
    state->external_id = request.external_id;
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

std::vector<UserOutput> OutPreprocessor::process_outputs(const std::map<uint64_t, EngineCoreOutput>& core_outputs)
{
    std::vector<UserOutput> user_outputs;
    user_outputs.reserve(core_outputs.size());

    for (const auto& item : core_outputs)
    {
        const EngineCoreOutput& core = item.second;

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
        out.external_id = state.external_id;

        if (state.is_finished)
        {
            out.text = state.cached_text;
            out.generated_token_ids = state.generated_token_ids;
            out.is_finished = true;
            out.finish_reason = state.finish_reason;
            user_outputs.push_back(std::move(out));
            continue;
        }

        if (core.has_error)
        {
            state.is_finished = true;
            state.finish_reason = "error";
            out.error_message = core.error_message;
            if (core.sequence != nullptr)
            {
                core.sequence->finished = true;
            }
        }
        else
        {
            out.delta_text = incremental_decode(state, core.new_token_id);
            if (check_stop_criteria(state, core.new_token_id) && core.sequence != nullptr)
            {
                core.sequence->finished = true;
            }
        }

        out.text = state.cached_text;
        out.generated_token_ids = state.generated_token_ids;
        out.is_finished = state.is_finished;
        out.finish_reason = state.finish_reason;
        user_outputs.push_back(std::move(out));
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

    if (latest_token == tokenizer->eos_id())
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