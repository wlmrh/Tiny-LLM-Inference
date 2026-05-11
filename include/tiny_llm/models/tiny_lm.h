#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tiny_llm/models/model.h"

namespace tiny_llm {

/**
 * @brief Tiny LM config loaded from a checkpoint.
 */
struct TinyLMConfig {
    int32_t num_layers = 1;
    int32_t vocab = 0;
    int32_t hidden = 0;
    int32_t bos_id = 1;
    int32_t eos_id = 2;
    int32_t unk_id = 3;
};

/**
 * @brief Small embedding + projection LM loaded from a text checkpoint.
 */
class TinyEmbeddingLM final : public Model {
public:
    TinyEmbeddingLM(TinyLMConfig cfg,
                    std::vector<float> embedding,
                    std::vector<float> projection,
                    std::vector<float> bias);

    /**
     * @brief Loads checkpoint from disk.
     */
    static TinyEmbeddingLM from_checkpoint(const std::string& path);

    const TinyLMConfig& config() const { return cfg_; }

    int32_t num_layers() const override { return cfg_.num_layers; }
    int32_t vocab_size() const override { return cfg_.vocab; }
    int32_t expected_bos_id() const override { return cfg_.bos_id; }
    int32_t expected_eos_id() const override { return cfg_.eos_id; }
    int32_t expected_unk_id() const override { return cfg_.unk_id; }

    Tensor forward(const PreparedInputs& inputs, RuntimeContext& ctx) override;

private:
    TinyLMConfig cfg_;
    std::vector<float> embedding_;
    std::vector<float> projection_;
    std::vector<float> bias_;
};

} // namespace tiny_llm
