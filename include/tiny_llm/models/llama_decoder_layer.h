#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/models/llama_config.h"
#include "tiny_llm/models/llama_weight_map.h"
#include "tiny_llm/models/modules/linear.h"
#include "tiny_llm/models/modules/rmsnorm.h"

namespace tiny_llm {

class ExecutionContext;

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

class LlamaSelfAttention {
public:
    explicit LlamaSelfAttention(const LlamaConfig& config);

    void load_weights(const WeightMap& weight_map, const std::string& prefix);
    void forward(const Tensor& hidden_states,
                 const Tensor& positions,
                 LlamaAttentionBuffers& buffers,
                 ExecutionContext& ctx) const;

    void validate_forward_inputs(const Tensor& hidden_states,
                                 const Tensor& positions,
                                 const LlamaAttentionBuffers& buffers) const;
private:
    void split_qkv(const Tensor& qkv, Tensor& q, Tensor& k, Tensor& v) const;
    void apply_rope(const Tensor& positions, Tensor& q, Tensor& k) const;
    void combine_qkv(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& combined) const;

    LlamaConfig config_;
    std::array<modules::StackedWeightDesc, 3> qkv_descs_{};
    modules::Linear qkv_proj_;
    modules::Linear o_proj_;
};

class LlamaMLP {
public:
    explicit LlamaMLP(const LlamaConfig& config);

    void load_weights(const WeightMap& weight_map, const std::string& prefix);
    void forward(const Tensor& hidden_states,
                 LlamaMLPBuffers& buffers,
                 ExecutionContext& ctx) const;

    void validate_forward_inputs(const Tensor& hidden_states,
                                 const LlamaMLPBuffers& buffers) const;
private:
    void apply_activation(const Tensor& gate, const Tensor& up, Tensor& activated) const;

    LlamaConfig config_;
    modules::Linear gate_proj_;
    modules::Linear up_proj_;
    modules::Linear down_proj_;
};

class LlamaDecoderLayer {
public:
    explicit LlamaDecoderLayer(const LlamaConfig& config);

    void load_weights(const WeightMap& weight_map, int layer_id);
    void forward(Tensor& hidden_states,
                 const Tensor& positions,
                 LlamaDecoderLayerBuffers& buffers,
                 ExecutionContext& ctx) const;

private:
    void validate_forward_inputs(const Tensor& hidden_states,
                                 const Tensor& positions,
                                 const LlamaDecoderLayerBuffers& buffers) const;
    void copy_tensor(const Tensor& src, Tensor& dst) const;
    void add_inplace(const Tensor& residual, const Tensor& update, Tensor& output) const;

    LlamaConfig config_;
    int32_t layer_id_ = -1;
    modules::RMSNorm input_layernorm_;
    LlamaSelfAttention self_attn_;
    modules::RMSNorm post_attention_layernorm_;
    LlamaMLP mlp_;
};

} // namespace tiny_llm
