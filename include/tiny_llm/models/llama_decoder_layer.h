#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/models/llama_config.h"
#include "tiny_llm/models/llama_weight_map.h"
#include "tiny_llm/models/modules/linear.h"
#include "tiny_llm/models/modules/rmsnorm.h"
#include "tiny_llm/models/modules/rotary_embedding.h"
#include "tiny_llm/runtime/runtime_context.h"

namespace tiny_llm {

struct LlamaAttentionBuffers {
    Tensor qkv;
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor attn_input;
    Tensor attn_output;
    Tensor proj_output;
};

struct LlamaMLPBuffers {
    Tensor gate_up;
    Tensor gate;
    Tensor up;
    Tensor activated;
    Tensor down;
};

struct LlamaDecoderLayerBuffers {
    Tensor residual;
    Tensor norm_output;
    LlamaAttentionBuffers attention;
    LlamaMLPBuffers mlp;
};

class LlamaSelfAttention : public torch::nn::Module {
public:
    explicit LlamaSelfAttention(const LlamaConfig& config);

    void load_weights(const WeightMap& weight_map, const std::string& prefix, int32_t layer_id);
    void forward(const Tensor& hidden_states,
                 const Tensor& positions,
                 LlamaAttentionBuffers& buffers,
                 RuntimeContext& ctx) const;

    void validate_forward_inputs(const Tensor& hidden_states,
                                 const Tensor& positions,
                                 const LlamaAttentionBuffers& buffers) const;
private:
    void split_qkv(const Tensor& qkv, Tensor& q, Tensor& k, Tensor& v) const;
    void apply_rope(const Tensor& positions, Tensor& q, Tensor& k) const;
    void compute_attention(const Tensor& positions,
                           const Tensor& q,
                           const Tensor& k,
                           const Tensor& v,
                           Tensor& out,
                           RuntimeContext& ctx) const;

    LlamaConfig config_;
    int32_t layer_id_ = -1;
    std::array<modules::StackedWeightDesc, 3> qkv_descs_{};
    std::shared_ptr<modules::Linear> qkv_proj_;
    std::shared_ptr<modules::Linear> o_proj_;
    std::shared_ptr<modules::RotaryEmbedding> rotary_;
};

class LlamaMLP : public torch::nn::Module {
public:
    explicit LlamaMLP(const LlamaConfig& config);

    void load_weights(const WeightMap& weight_map, const std::string& prefix);
    void forward(const Tensor& hidden_states,
                 LlamaMLPBuffers& buffers,
                 RuntimeContext& ctx) const;

    void validate_forward_inputs(const Tensor& hidden_states,
                                 const LlamaMLPBuffers& buffers) const;
private:
    void apply_activation(const Tensor& gate, const Tensor& up, Tensor& activated) const;

    LlamaConfig config_;
    std::array<modules::StackedWeightDesc, 2> gate_up_descs_{};
    std::shared_ptr<modules::Linear> gate_up_proj_;
    std::shared_ptr<modules::Linear> down_proj_;
};

class LlamaDecoderLayer : public torch::nn::Module {
public:
    explicit LlamaDecoderLayer(const LlamaConfig& config);

    void load_weights(const WeightMap& weight_map, int layer_id);
    void forward(Tensor& hidden_states,
                 const Tensor& positions,
                 LlamaDecoderLayerBuffers& buffers,
                 RuntimeContext& ctx) const;

private:
    void validate_forward_inputs(const Tensor& hidden_states,
                                 const Tensor& positions,
                                 const LlamaDecoderLayerBuffers& buffers) const;
    void copy_tensor(const Tensor& src, Tensor& dst) const;
    void add_inplace(const Tensor& residual, const Tensor& update, Tensor& output) const;

    LlamaConfig config_;
    int32_t layer_id_ = -1;
    std::shared_ptr<modules::RMSNorm> input_layernorm_;
    std::shared_ptr<LlamaSelfAttention> self_attn_;
    std::shared_ptr<modules::RMSNorm> post_attention_layernorm_;
    std::shared_ptr<LlamaMLP> mlp_;
};

} // namespace tiny_llm
