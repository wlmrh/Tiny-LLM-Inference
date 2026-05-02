#include "tiny_llm/models/llama_decoder_layer.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/operators/llama_ops.h"
#include "tiny_llm/operators/paged_attention.h"
#include "tiny_llm/runtime/kv_cache.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tiny_llm {

namespace {

int32_t kv_hidden_size(const LlamaConfig& config)
{
    return config.num_key_value_heads * config.head_dim;
}

void validate_cpu_tensor(const Tensor& tensor, const char* name)
{
    if (tensor.device().is_cuda())
    {
        throw std::runtime_error(std::string(name) + ": CUDA tensor path is not implemented for this helper.");
    }
}

void validate_float_tensor_2d(const Tensor& tensor,
                              int64_t rows,
                              int64_t cols,
                              const char* name)
{
    if (!tensor.defined())
    {
        throw std::runtime_error(std::string(name) + ": tensor must be defined.");
    }
    if (tensor_dtype(tensor) != DType::kFloat32)
    {
        throw std::runtime_error(std::string(name) + ": tensor must be float32.");
    }
    if (tensor.dim() != 2 || tensor.size(0) != rows || tensor.size(1) != cols)
    {
        throw std::runtime_error(std::string(name) + ": unexpected shape.");
    }
    if (tensor_data(tensor) == nullptr)
    {
        throw std::runtime_error(std::string(name) + ": tensor data pointer must be non-null.");
    }
}

void validate_int_tensor_1d(const Tensor& tensor, int64_t size, const char* name)
{
    if (!tensor.defined())
    {
        throw std::runtime_error(std::string(name) + ": tensor must be defined.");
    }
    if (tensor_dtype(tensor) != DType::kInt32)
    {
        throw std::runtime_error(std::string(name) + ": tensor must be int32.");
    }
    if (tensor.dim() != 1 || tensor.size(0) != size)
    {
        throw std::runtime_error(std::string(name) + ": unexpected shape.");
    }
    if (tensor_data(tensor) == nullptr)
    {
        throw std::runtime_error(std::string(name) + ": tensor data pointer must be non-null.");
    }
}

} // namespace

LlamaSelfAttention::LlamaSelfAttention(const LlamaConfig& config)
    : config_(config),
      qkv_proj_(config.hidden_size, config.hidden_size + 2 * kv_hidden_size(config)),
      o_proj_(config.hidden_size, config.hidden_size)
{
}

void LlamaSelfAttention::load_weights(const WeightMap& weight_map, const std::string& prefix, int32_t layer_id)
{
    if (layer_id < 0)
    {
        throw std::runtime_error("LlamaSelfAttention::load_weights: layer_id must be non-negative.");
    }
    layer_id_ = layer_id;
    qkv_descs_[0] = {
        weight_map.get_tensor_as<float>(prefix + "self_attn.q_proj.weight"),
        config_.hidden_size,
        config_.hidden_size,
        0,
        modules::WeightLayout::kOutIn,
    };
    qkv_descs_[1] = {
        weight_map.get_tensor_as<float>(prefix + "self_attn.k_proj.weight"),
        kv_hidden_size(config_),
        config_.hidden_size,
        config_.hidden_size,
        modules::WeightLayout::kOutIn,
    };
    qkv_descs_[2] = {
        weight_map.get_tensor_as<float>(prefix + "self_attn.v_proj.weight"),
        kv_hidden_size(config_),
        config_.hidden_size,
        config_.hidden_size + kv_hidden_size(config_),
        modules::WeightLayout::kOutIn,
    };
    qkv_proj_.bind_stacked_weights(qkv_descs_.data(), static_cast<int32_t>(qkv_descs_.size()));
    o_proj_.bind_weight(
        weight_map.get_tensor_as<float>(prefix + "self_attn.o_proj.weight"),
        config_.hidden_size,
        config_.hidden_size,
        modules::WeightLayout::kOutIn);
}

void LlamaSelfAttention::forward(const Tensor& hidden_states,
                                 const Tensor& positions,
                                 LlamaAttentionBuffers& buffers,
                                 ExecutionContext& ctx) const
{
    validate_forward_inputs(hidden_states, positions, buffers);

    qkv_proj_.forward(hidden_states, buffers.qkv, ctx);
    split_qkv(buffers.qkv, buffers.q, buffers.k, buffers.v);
    apply_rope(positions, buffers.q, buffers.k);
    compute_attention(positions, buffers.q, buffers.k, buffers.v, buffers.attn_output, ctx);
    o_proj_.forward(buffers.attn_output, buffers.proj_output, ctx);
}

void LlamaSelfAttention::validate_forward_inputs(const Tensor& hidden_states,
                                                 const Tensor& positions,
                                                 const LlamaAttentionBuffers& buffers) const
{
    const int64_t rows = hidden_states.size(0);
    validate_float_tensor_2d(hidden_states, rows, config_.hidden_size, "LlamaSelfAttention::hidden_states");
    validate_int_tensor_1d(positions, rows, "LlamaSelfAttention::positions");
    validate_float_tensor_2d(
        buffers.qkv,
        rows,
        config_.hidden_size + 2 * kv_hidden_size(config_),
        "LlamaSelfAttention::qkv");
    validate_float_tensor_2d(buffers.q, rows, config_.hidden_size, "LlamaSelfAttention::q");
    validate_float_tensor_2d(buffers.k, rows, kv_hidden_size(config_), "LlamaSelfAttention::k");
    validate_float_tensor_2d(buffers.v, rows, kv_hidden_size(config_), "LlamaSelfAttention::v");
    validate_float_tensor_2d(buffers.attn_input, rows, config_.hidden_size, "LlamaSelfAttention::attn_input");
    validate_float_tensor_2d(buffers.attn_output, rows, config_.hidden_size, "LlamaSelfAttention::attn_output");
    validate_float_tensor_2d(buffers.proj_output, rows, config_.hidden_size, "LlamaSelfAttention::proj_output");
}

void LlamaSelfAttention::split_qkv(const Tensor& qkv, Tensor& q, Tensor& k, Tensor& v) const
{
    ops::split_qkv(qkv, q, k, v, config_.hidden_size, kv_hidden_size(config_));
}

void LlamaSelfAttention::apply_rope(const Tensor& positions, Tensor& q, Tensor& k) const
{
    ops::apply_rope(
        positions,
        q,
        k,
        config_.num_attention_heads,
        config_.num_key_value_heads,
        config_.head_dim,
        config_.rope_theta);
}

void LlamaSelfAttention::compute_attention(const Tensor& positions,
                                           const Tensor& q,
                                           const Tensor& k,
                                           const Tensor& v,
                                           Tensor& out,
                                           ExecutionContext& ctx) const
{
    ops::llama_attention(
        positions,
        q,
        k,
        v,
        out,
        ctx,
        layer_id_,
        config_.num_attention_heads,
        config_.num_key_value_heads,
        config_.head_dim);
}

LlamaMLP::LlamaMLP(const LlamaConfig& config)
    : config_(config),
      gate_proj_(config.hidden_size, config.intermediate_size),
      up_proj_(config.hidden_size, config.intermediate_size),
      down_proj_(config.intermediate_size, config.hidden_size)
{
    if (config_.hidden_act != "silu")
    {
        throw std::runtime_error("LlamaMLP: only silu hidden_act is supported in Phase 3.");
    }
}

void LlamaMLP::load_weights(const WeightMap& weight_map, const std::string& prefix)
{
    gate_proj_.bind_weight(
        weight_map.get_tensor_as<float>(prefix + "mlp.gate_proj.weight"),
        config_.intermediate_size,
        config_.hidden_size,
        modules::WeightLayout::kOutIn);
    up_proj_.bind_weight(
        weight_map.get_tensor_as<float>(prefix + "mlp.up_proj.weight"),
        config_.intermediate_size,
        config_.hidden_size,
        modules::WeightLayout::kOutIn);
    down_proj_.bind_weight(
        weight_map.get_tensor_as<float>(prefix + "mlp.down_proj.weight"),
        config_.hidden_size,
        config_.intermediate_size,
        modules::WeightLayout::kOutIn);
}

void LlamaMLP::forward(const Tensor& hidden_states,
                       LlamaMLPBuffers& buffers,
                       ExecutionContext& ctx) const
{
    validate_forward_inputs(hidden_states, buffers);
    gate_proj_.forward(hidden_states, buffers.gate, ctx);
    up_proj_.forward(hidden_states, buffers.up, ctx);
    apply_activation(buffers.gate, buffers.up, buffers.activated);
    down_proj_.forward(buffers.activated, buffers.down, ctx);
}

void LlamaMLP::validate_forward_inputs(const Tensor& hidden_states,
                                       const LlamaMLPBuffers& buffers) const
{
    const int64_t rows = hidden_states.size(0);
    validate_float_tensor_2d(hidden_states, rows, config_.hidden_size, "LlamaMLP::hidden_states");
    validate_float_tensor_2d(buffers.gate, rows, config_.intermediate_size, "LlamaMLP::gate");
    validate_float_tensor_2d(buffers.up, rows, config_.intermediate_size, "LlamaMLP::up");
    validate_float_tensor_2d(buffers.activated, rows, config_.intermediate_size, "LlamaMLP::activated");
    validate_float_tensor_2d(buffers.down, rows, config_.hidden_size, "LlamaMLP::down");
}

void LlamaMLP::apply_activation(const Tensor& gate, const Tensor& up, Tensor& activated) const
{
    ops::silu_multiply(gate, up, activated);
}

LlamaDecoderLayer::LlamaDecoderLayer(const LlamaConfig& config)
    : config_(config),
      input_layernorm_(config.hidden_size, config.rms_norm_eps),
      self_attn_(config),
      post_attention_layernorm_(config.hidden_size, config.rms_norm_eps),
      mlp_(config)
{
}

void LlamaDecoderLayer::load_weights(const WeightMap& weight_map, int layer_id)
{
    if (layer_id < 0)
    {
        throw std::runtime_error("LlamaDecoderLayer::load_weights: layer_id must be non-negative.");
    }

    layer_id_ = layer_id;
    const std::string prefix = "model.layers." + std::to_string(layer_id) + ".";
    input_layernorm_.bind_weights(weight_map.get_tensor_as<float>(prefix + "input_layernorm.weight"));
    self_attn_.load_weights(weight_map, prefix, layer_id);
    post_attention_layernorm_.bind_weights(weight_map.get_tensor_as<float>(prefix + "post_attention_layernorm.weight"));
    mlp_.load_weights(weight_map, prefix);
}

void LlamaDecoderLayer::forward(Tensor& hidden_states,
                                const Tensor& positions,
                                LlamaDecoderLayerBuffers& buffers,
                                ExecutionContext& ctx) const
{
    validate_forward_inputs(hidden_states, positions, buffers);

    copy_tensor(hidden_states, buffers.residual);
    input_layernorm_.forward(hidden_states, buffers.norm_output, ctx);
    self_attn_.forward(buffers.norm_output, positions, buffers.attention, ctx);
    add_inplace(buffers.residual, buffers.attention.proj_output, hidden_states);

    copy_tensor(hidden_states, buffers.residual);
    post_attention_layernorm_.forward(hidden_states, buffers.norm_output, ctx);
    mlp_.forward(buffers.norm_output, buffers.mlp, ctx);
    add_inplace(buffers.residual, buffers.mlp.down, hidden_states);
}

void LlamaDecoderLayer::validate_forward_inputs(const Tensor& hidden_states,
                                                const Tensor& positions,
                                                const LlamaDecoderLayerBuffers& buffers) const
{
    if (layer_id_ < 0)
    {
        throw std::runtime_error("LlamaDecoderLayer::forward: weights must be loaded before forward.");
    }

    const int64_t rows = hidden_states.size(0);
    validate_float_tensor_2d(hidden_states, rows, config_.hidden_size, "LlamaDecoderLayer::hidden_states");
    validate_int_tensor_1d(positions, rows, "LlamaDecoderLayer::positions");
    validate_float_tensor_2d(buffers.residual, rows, config_.hidden_size, "LlamaDecoderLayer::residual");
    validate_float_tensor_2d(buffers.norm_output, rows, config_.hidden_size, "LlamaDecoderLayer::norm_output");
    self_attn_.validate_forward_inputs(buffers.norm_output, positions, buffers.attention);
    mlp_.validate_forward_inputs(buffers.norm_output, buffers.mlp);
}

void LlamaDecoderLayer::copy_tensor(const Tensor& src, Tensor& dst) const
{
    ops::copy_tensor(src, dst);
}

void LlamaDecoderLayer::add_inplace(const Tensor& residual, const Tensor& update, Tensor& output) const
{
    ops::add_tensors(residual, update, output);
}

} // namespace tiny_llm
