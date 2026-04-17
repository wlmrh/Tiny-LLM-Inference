#pragma once

#include <cstdint>

#include "tiny_llm/models/model.h"

namespace tiny_llm {

class ExecutionContext;
class Tensor;

/**
 * @brief Compact model configuration used by the MiniLLaMA skeleton.
 */
struct MiniLLaMAConfig {
    /// Number of transformer decoder layers.
    int32_t num_layers = 4;
    /// Hidden size of model states.
    int32_t hidden = 512;
    /// Number of attention heads.
    int32_t num_heads = 8;
    /// Per-head hidden dimension.
    int32_t head_dim = 64;
    /// Vocabulary size for output logits.
    int32_t vocab = 32000;
};

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

    int32_t num_layers() const override { return cfg_.num_layers; }

    int32_t vocab_size() const override { return cfg_.vocab; }

    /**
     * @brief Run one decoding step and write logits for next-token sampling.
     */
    void forward_step(const Tensor& input_ids,
                      const Tensor& positions,
                      Tensor& logits,
                      ExecutionContext& ctx) override;

private:
    /// Static model hyper-parameters.
    MiniLLaMAConfig cfg_;
};

} // namespace tiny_llm
