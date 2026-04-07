#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/engine_core.h"
#include "tiny_llm/runtime/processors.h"
#include "tiny_llm/runtime/scheduler.h"

namespace tiny_llm {

class ExecutionContext;
class KVCache;
class Model;
class Tokenizer;

/**
 * @brief Frontend wrapper: handles strings/tokenizer/output over token-only EngineCore.
 */
class LLMEngine {
public:
    explicit LLMEngine(const EngineArgs& args);

    /**
     * @brief Adds one text request and returns internal request id.
     */
    uint64_t add_request(const std::string& prompt,
                         const UserSamplingParams& user_params = UserSamplingParams{},
                         const std::string& ext_request_id = "");

    /**
     * @brief Returns true when any request is still unfinished.
     */
    bool has_unfinished_requests() const;

    /**
     * @brief Runs one decode step for active requests and returns user outputs.
     */
    std::vector<UserOutput> step();

private:
    std::unique_ptr<EngineCore> core_;
    InputPreprocessor input_preprocessor_;
    OutPreprocessor output_preprocessor_;
};

} // namespace tiny_llm
