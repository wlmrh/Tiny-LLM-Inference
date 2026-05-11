#pragma once

#include <cstdint>

#include "tiny_llm/models/llama_config.h"
#include "tiny_llm/models/model.h"

namespace tiny_llm {

class ExecutionContext;

/**
 * @brief Minimal model wrapper exposing a single-token forward API.
 */
class MiniLLaMA : public Model {
public:
    explicit MiniLLaMA(MiniLLaMAConfig cfg) : cfg_(cfg) {}

    /**
     * @brief Returns model hyper-parameters.
     */
    const MiniLLaMAConfig& config() const { return cfg_; }

    int32_t num_layers() const override { return cfg_.num_hidden_layers; }

    int32_t vocab_size() const override { return cfg_.vocab_size; }

    int32_t expected_bos_id() const override { return cfg_.bos_token_id; }

    int32_t expected_eos_id() const override { return cfg_.eos_token_id; }

    int32_t expected_unk_id() const override { return cfg_.unk_token_id; }

    Tensor forward(const PreparedInputs& inputs, RuntimeContext& ctx) override;

private:
    /// Static model hyper-parameters.
    MiniLLaMAConfig cfg_;
};

} // namespace tiny_llm
