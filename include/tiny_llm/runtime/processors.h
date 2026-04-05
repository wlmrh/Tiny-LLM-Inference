#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tiny_llm {

class Model;
class Tokenizer;
class TokenizerRegistry;
struct CoreSequence;

/**
 * @brief User-facing sampling configuration.
 */
struct UserSamplingParams {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int32_t top_k = 0;
    int32_t max_tokens = 32;
    std::vector<int32_t> stop_token_ids;
};

/**
 * @brief Engine-ready normalized sampling configuration.
 */
struct SamplingParams {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int32_t top_k = 0;
    int32_t max_tokens = 32;
    std::vector<int32_t> stop_token_ids;
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
    int32_t generated_tokens = 0;
    CoreSequence* sequence = nullptr;
    bool has_error = false;
    std::string error_message;
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
 * @brief Stateless input translator and validator.
 */
class InputPreprocessor {
public:
    InputPreprocessor(Tokenizer* tokenizer,
                      const Model* model,
                      int32_t default_max_tokens);

    InputPreprocessor(TokenizerRegistry* tokenizer_registry,
                      const Model* model,
                      int32_t default_max_tokens);

    EngineCoreRequest process_inputs(const std::string& prompt,
                                     const UserSamplingParams& user_params,
                                     const std::string& ext_request_id) const;

private:
    uint64_t assign_internal_id() const;
    std::vector<int32_t> tokenize(const std::string& text) const;
    std::string apply_chat_template(const std::string& text) const;
    SamplingParams normalize_sampling_params(const UserSamplingParams& user_params) const;
    void bind_external_id(EngineCoreRequest& request, const std::string& ext_request_id) const;
    void validate_model_tokenizer_contract() const;
    void validate_prompt_tokens(const std::vector<int32_t>& token_ids) const;
    void validate_sampling_params(const SamplingParams& sampling_params) const;

    Tokenizer* tokenizer_ = nullptr;
    const Model* model_ = nullptr;
    int32_t default_max_tokens_ = 32;
    mutable uint64_t next_internal_id_ = 1;
    mutable std::unordered_map<std::string, uint64_t> external_to_internal_id_;
};

/**
 * @brief Stateful output assembler and termination checker.
 */
class OutPreprocessor {
public:
    explicit OutPreprocessor(const Tokenizer* tokenizer);
    explicit OutPreprocessor(const TokenizerRegistry* tokenizer_registry);

    void add_request(const EngineCoreRequest& request);
    std::vector<UserOutput> process_outputs(const std::map<uint64_t, EngineCoreOutput>& core_outputs);
    bool has_unfinished_requests() const;

private:
    std::string incremental_decode(RequestState& state, int32_t new_token_id);
    bool check_stop_criteria(RequestState& state, int32_t latest_token);

    const Tokenizer* tokenizer_ = nullptr;
    std::unordered_map<uint64_t, std::unique_ptr<RequestState>> states_;
};

} // namespace tiny_llm