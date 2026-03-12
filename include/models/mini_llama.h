#pragma once
#include <cstdint>

namespace tiny_llm {

class ExecutionContext;
class Tensor;

struct MiniLLaMAConfig {
    int32_t num_layers = 4;
    int32_t hidden = 512;
    int32_t num_heads = 8;
    int32_t head_dim = 64;
    int32_t vocab = 32000;
};

class MiniLLaMA {
public:
    explicit MiniLLaMA(MiniLLaMAConfig cfg) : cfg_(cfg) {}

    // 初版：single-step（1 token）forward
    void forward_step(const Tensor& input_ids,
                      const Tensor& positions,
                      Tensor& logits,
                      ExecutionContext& ctx);

private:
    MiniLLaMAConfig cfg_;
    // weights handle omitted for now
};

} // namespace tiny_llm