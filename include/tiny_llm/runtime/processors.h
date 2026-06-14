#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tiny_llm {

class Tokenizer;
struct EngineArgs;

/**
 * @brief Sampling fields shared by user-facing and normalized engine parameters.
 */
struct SamplingParamsCommon {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int32_t top_k = 0;
    float repetition_penalty = 1.0f;
    uint64_t seed = 0;
    bool ignore_eos = false;
    std::vector<int32_t> stop_token_ids;
};

/**
 * @brief User-facing sampling configuration.
 *
 * A max_tokens value of 0 means the runtime default should be used.
 */
struct UserSamplingParams : public SamplingParamsCommon {
    int32_t max_tokens = 0;
};

/**
 * @brief Engine-ready normalized sampling configuration.
 */
struct SamplingParams : public SamplingParamsCommon {
    int32_t max_tokens = 32;
};

/**
 * @brief Input protocol object consumed by scheduler and model executor.
 */
struct EngineCoreRequest {
    uint64_t internal_id = 0;
    std::string external_id;
    std::vector<int32_t> prompt_token_ids;
    SamplingParams sampling_params;
};

/**
 * @brief Core output message produced per decode step.
 */
struct EngineCoreOutput {
    uint64_t internal_id = 0;
    int32_t new_token_id = -1;
};
/**
 * @brief User-facing output emitted by OutPreprocessor.
 */
struct UserOutput {
    uint64_t internal_id = 0;
    std::string external_id;
    std::string delta_text;
    std::string text;
    std::vector<int32_t> generated_token_ids;
    bool is_finished = false;
    std::string finish_reason;
    std::string error_message;
};

/**
 * @brief Mutable request state managed by OutPreprocessor.
 */
struct RequestState {
    uint64_t internal_id = 0;
    std::string external_id;
    SamplingParams sampling_params;
    std::vector<int32_t> prompt_token_ids;
    std::vector<int32_t> generated_token_ids;
    size_t decoded_prefix_len = 0;
    bool is_finished = false;
    std::string finish_reason;
    std::string cached_text;
};

/**
 * @brief Stateful input translator, request ID binder, and validator.
 */
class InputPreprocessor {
public:
    explicit InputPreprocessor(const EngineArgs& args);

    EngineCoreRequest process_inputs(const std::string& prompt,
                                     const UserSamplingParams& user_params,
                                     const std::string& ext_request_id);
    void release_request(const std::string& external_id, uint64_t internal_id);

private:
    uint64_t assign_internal_id();
    std::vector<int32_t> tokenize(const std::string& text) const;
    SamplingParams normalize_sampling_params(const UserSamplingParams& user_params) const;
    void bind_external_id(EngineCoreRequest& request, const std::string& ext_request_id);
    void validate_tokenizer_contract() const;
    void validate_prompt_tokens(const std::vector<int32_t>& token_ids) const;
    void validate_sampling_params(const SamplingParams& sampling_params) const;

    Tokenizer* tokenizer_ = nullptr;
    int32_t default_max_tokens_ = 32;
    uint64_t next_internal_id_ = 1;
    std::unordered_map<std::string, uint64_t> external_to_internal_id_;
};

/**
 * @brief Stateful output assembler and termination checker.
 */
class OutPreprocessor {
public:
    explicit OutPreprocessor(const EngineArgs& args);

    void add_request(const EngineCoreRequest& request);
    std::vector<UserOutput> process_outputs(const std::vector<EngineCoreOutput>& core_outputs);
    bool has_unfinished_requests() const;

private:
    std::string incremental_decode(RequestState& state, int32_t new_token_id);
    bool check_stop_criteria(RequestState& state, int32_t latest_token);

    const Tokenizer* tokenizer_ = nullptr;
    std::unordered_map<uint64_t, std::unique_ptr<RequestState>> states_;
};

} // namespace tiny_llm
