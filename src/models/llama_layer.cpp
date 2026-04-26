#include "tiny_llm/models/llama_decoder_layer.h"

#include "tiny_llm/core/context.h"
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
    validate_cpu_tensor(qkv, "LlamaSelfAttention::split_qkv");
    const int64_t rows = qkv.size(0);
    const float* qkv_ptr = static_cast<const float*>(tensor_data(qkv));
    float* q_ptr = static_cast<float*>(tensor_data(q));
    float* k_ptr = static_cast<float*>(tensor_data(k));
    float* v_ptr = static_cast<float*>(tensor_data(v));

    for (int64_t row = 0; row < rows; ++row)
    {
        const size_t qkv_offset =
            static_cast<size_t>(row) * static_cast<size_t>(config_.hidden_size + 2 * kv_hidden_size(config_));
        const size_t out_offset = static_cast<size_t>(row) * static_cast<size_t>(config_.hidden_size);
        const size_t kv_out_offset = static_cast<size_t>(row) * static_cast<size_t>(kv_hidden_size(config_));
        for (int32_t col = 0; col < config_.hidden_size; ++col)
        {
            q_ptr[out_offset + static_cast<size_t>(col)] = qkv_ptr[qkv_offset + static_cast<size_t>(col)];
        }
        for (int32_t col = 0; col < kv_hidden_size(config_); ++col)
        {
            k_ptr[kv_out_offset + static_cast<size_t>(col)] =
                qkv_ptr[qkv_offset + static_cast<size_t>(config_.hidden_size + col)];
            v_ptr[kv_out_offset + static_cast<size_t>(col)] =
                qkv_ptr[qkv_offset + static_cast<size_t>(config_.hidden_size + kv_hidden_size(config_) + col)];
        }
    }
}

void LlamaSelfAttention::apply_rope(const Tensor& positions, Tensor& q, Tensor& k) const
{
    validate_cpu_tensor(positions, "LlamaSelfAttention::apply_rope");
    validate_cpu_tensor(q, "LlamaSelfAttention::apply_rope");
    validate_cpu_tensor(k, "LlamaSelfAttention::apply_rope");

    const int32_t* positions_ptr = static_cast<const int32_t*>(tensor_data(positions));
    float* q_ptr = static_cast<float*>(tensor_data(q));
    float* k_ptr = static_cast<float*>(tensor_data(k));
    const int64_t rows = q.size(0);

    for (int64_t row = 0; row < rows; ++row)
    {
        const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(config_.hidden_size);
        const float position = static_cast<float>(positions_ptr[row]);
        for (int32_t head = 0; head < config_.num_attention_heads; ++head)
        {
            const int32_t head_offset = head * config_.head_dim;
            const int32_t rotary_half = config_.head_dim / 2;
            for (int32_t dim = 0; dim < rotary_half; ++dim)
            {
                const int32_t idx0 = head_offset + dim;
                const int32_t idx1 = head_offset + rotary_half + dim;
                const float theta = position / std::pow(
                    config_.rope_theta,
                    static_cast<float>(2 * dim) / static_cast<float>(config_.head_dim));
                const float cos_theta = std::cos(theta);
                const float sin_theta = std::sin(theta);

                const size_t q0 = row_offset + static_cast<size_t>(idx0);
                const size_t q1 = row_offset + static_cast<size_t>(idx1);
                const float qv0 = q_ptr[q0];
                const float qv1 = q_ptr[q1];
                q_ptr[q0] = qv0 * cos_theta - qv1 * sin_theta;
                q_ptr[q1] = qv1 * cos_theta + qv0 * sin_theta;
            }
        }

        const size_t kv_row_offset = static_cast<size_t>(row) * static_cast<size_t>(kv_hidden_size(config_));
        for (int32_t head = 0; head < config_.num_key_value_heads; ++head)
        {
            const int32_t head_offset = head * config_.head_dim;
            const int32_t rotary_half = config_.head_dim / 2;
            for (int32_t dim = 0; dim < rotary_half; ++dim)
            {
                const int32_t idx0 = head_offset + dim;
                const int32_t idx1 = head_offset + rotary_half + dim;
                const float theta = position / std::pow(
                    config_.rope_theta,
                    static_cast<float>(2 * dim) / static_cast<float>(config_.head_dim));
                const float cos_theta = std::cos(theta);
                const float sin_theta = std::sin(theta);

                const size_t k0 = kv_row_offset + static_cast<size_t>(idx0);
                const size_t k1 = kv_row_offset + static_cast<size_t>(idx1);
                const float kv0 = k_ptr[k0];
                const float kv1 = k_ptr[k1];
                k_ptr[k0] = kv0 * cos_theta - kv1 * sin_theta;
                k_ptr[k1] = kv1 * cos_theta + kv0 * sin_theta;
            }
        }
    }
}

void LlamaSelfAttention::compute_attention(const Tensor& positions,
                                           const Tensor& q,
                                           const Tensor& k,
                                           const Tensor& v,
                                           Tensor& out,
                                           ExecutionContext& ctx) const
{
    validate_cpu_tensor(positions, "LlamaSelfAttention::compute_attention");
    validate_cpu_tensor(q, "LlamaSelfAttention::compute_attention");
    validate_cpu_tensor(k, "LlamaSelfAttention::compute_attention");
    validate_cpu_tensor(v, "LlamaSelfAttention::compute_attention");
    validate_cpu_tensor(out, "LlamaSelfAttention::compute_attention");

    const int32_t* positions_ptr = static_cast<const int32_t*>(tensor_data(positions));
    const float* q_ptr = static_cast<const float*>(tensor_data(q));
    const float* k_ptr = static_cast<const float*>(tensor_data(k));
    const float* v_ptr = static_cast<const float*>(tensor_data(v));
    float* out_ptr = static_cast<float*>(tensor_data(out));

    const int32_t group_size = config_.num_attention_heads / config_.num_key_value_heads;
    const int64_t rows = q.size(0);
    const float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));

    const ops::PagedAttentionRuntimeMetadata& metadata = ops::current_paged_attention_runtime_metadata();
    KVCache* kv_cache = ctx.kv();
    if (metadata.enabled && kv_cache != nullptr)
    {
        if (layer_id_ < 0)
        {
            throw std::runtime_error("LlamaSelfAttention::compute_attention: layer id is not set.");
        }
        if (kv_cache->block_size_tokens() != metadata.block_size_tokens)
        {
            throw std::runtime_error("LlamaSelfAttention::compute_attention: KV block size mismatch.");
        }
        if (metadata.seq_indices == nullptr || metadata.context_lens == nullptr || metadata.block_tables == nullptr)
        {
            throw std::runtime_error("LlamaSelfAttention::compute_attention: paged metadata is incomplete.");
        }

        const Tensor& seq_indices = *metadata.seq_indices;
        const Tensor& context_lens = *metadata.context_lens;
        const Tensor& block_tables = *metadata.block_tables;
        const std::vector<int64_t> seq_shape = tensor_shape(seq_indices);
        const std::vector<int64_t> context_shape = tensor_shape(context_lens);
        const std::vector<int64_t> block_shape = tensor_shape(block_tables);
        if (seq_shape.size() != 1 || seq_shape[0] != rows)
        {
            throw std::runtime_error("LlamaSelfAttention::compute_attention: seq_indices shape mismatch.");
        }
        if (context_shape.size() != 1 || block_shape.size() != 3
            || block_shape[0] <= layer_id_ || block_shape[1] != context_shape[0])
        {
            throw std::runtime_error("LlamaSelfAttention::compute_attention: block table shape mismatch.");
        }

        const size_t kv_token_bytes =
            static_cast<size_t>(kv_hidden_size(config_)) * sizeof(float);
        const size_t required_block_bytes =
            2 * static_cast<size_t>(metadata.block_size_tokens) * kv_token_bytes;
        if (kv_cache->block_size_bytes() < required_block_bytes)
        {
            throw std::runtime_error("LlamaSelfAttention::compute_attention: KV block byte size is too small.");
        }

        const int32_t* seq_index_ptr = seq_indices.data_ptr<int32_t>();
        const int32_t* context_ptr = context_lens.data_ptr<int32_t>();
        const int32_t* block_ptr = block_tables.data_ptr<int32_t>();
        const int64_t num_layers = block_shape[0];
        const int64_t num_seqs = block_shape[1];
        const int64_t max_blocks_per_seq = block_shape[2];

        auto block_id_for = [&](int32_t seq_index, int32_t position) -> int32_t {
            if (seq_index < 0 || seq_index >= num_seqs)
            {
                throw std::runtime_error("LlamaSelfAttention::compute_attention: seq index out of range.");
            }
            const int32_t logical_block = position / metadata.block_size_tokens;
            if (logical_block < 0 || logical_block >= max_blocks_per_seq)
            {
                throw std::runtime_error("LlamaSelfAttention::compute_attention: logical block out of range.");
            }
            const int64_t index =
                static_cast<int64_t>(layer_id_) * num_seqs * max_blocks_per_seq
                + static_cast<int64_t>(seq_index) * max_blocks_per_seq
                + logical_block;
            (void)num_layers;
            const int32_t block_id = block_ptr[index];
            if (block_id < 0)
            {
                throw std::runtime_error("LlamaSelfAttention::compute_attention: missing physical KV block.");
            }
            return block_id;
        };

        auto k_block_ptr = [&](int32_t block_id) -> float* {
            void* block = kv_cache->block_ptr(block_id);
            if (block == nullptr)
            {
                throw std::runtime_error("LlamaSelfAttention::compute_attention: KV block pointer is null.");
            }
            return static_cast<float*>(block);
        };

        auto v_block_ptr = [&](int32_t block_id) -> float* {
            return k_block_ptr(block_id)
                + static_cast<size_t>(metadata.block_size_tokens) * static_cast<size_t>(kv_hidden_size(config_));
        };

        for (int64_t row = 0; row < rows; ++row)
        {
            const int32_t seq_index = seq_index_ptr[row];
            const int32_t position = positions_ptr[row];
            const int32_t block_id = block_id_for(seq_index, position);
            const int32_t token_offset = position % metadata.block_size_tokens;
            const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(kv_hidden_size(config_));
            float* key_dst = k_block_ptr(block_id)
                + static_cast<size_t>(token_offset) * static_cast<size_t>(kv_hidden_size(config_));
            float* value_dst = v_block_ptr(block_id)
                + static_cast<size_t>(token_offset) * static_cast<size_t>(kv_hidden_size(config_));
            std::memcpy(key_dst, k_ptr + row_offset, kv_token_bytes);
            std::memcpy(value_dst, v_ptr + row_offset, kv_token_bytes);
        }

        std::vector<float> scores;
        for (int64_t row = 0; row < rows; ++row)
        {
            const int32_t seq_index = seq_index_ptr[row];
            const int32_t target_position = positions_ptr[row];
            if (target_position < 0 || target_position >= context_ptr[seq_index])
            {
                throw std::runtime_error("LlamaSelfAttention::compute_attention: target position exceeds context length.");
            }

            const int32_t context_len = target_position + 1;
            scores.assign(static_cast<size_t>(context_len), -std::numeric_limits<float>::infinity());
            const size_t q_row_offset = static_cast<size_t>(row) * static_cast<size_t>(config_.hidden_size);
            for (int32_t q_head = 0; q_head < config_.num_attention_heads; ++q_head)
            {
                const int32_t kv_head = q_head / group_size;
                float max_score = -std::numeric_limits<float>::infinity();
                for (int32_t src_pos = 0; src_pos <= target_position; ++src_pos)
                {
                    const int32_t block_id = block_id_for(seq_index, src_pos);
                    const int32_t token_offset = src_pos % metadata.block_size_tokens;
                    const float* key_base = k_block_ptr(block_id)
                        + static_cast<size_t>(token_offset) * static_cast<size_t>(kv_hidden_size(config_));

                    float score = 0.0f;
                    for (int32_t dim = 0; dim < config_.head_dim; ++dim)
                    {
                        const size_t q_index =
                            q_row_offset + static_cast<size_t>(q_head * config_.head_dim + dim);
                        const size_t k_index =
                            static_cast<size_t>(kv_head * config_.head_dim + dim);
                        score += q_ptr[q_index] * key_base[k_index];
                    }
                    score *= scale;
                    scores[static_cast<size_t>(src_pos)] = score;
                    max_score = std::max(max_score, score);
                }

                float score_sum = 0.0f;
                for (int32_t src_pos = 0; src_pos <= target_position; ++src_pos)
                {
                    const float exp_score = std::exp(scores[static_cast<size_t>(src_pos)] - max_score);
                    scores[static_cast<size_t>(src_pos)] = exp_score;
                    score_sum += exp_score;
                }
                if (score_sum <= 0.0f)
                {
                    throw std::runtime_error("LlamaSelfAttention::compute_attention: no paged causal source tokens.");
                }

                for (int32_t dim = 0; dim < config_.head_dim; ++dim)
                {
                    float value = 0.0f;
                    for (int32_t src_pos = 0; src_pos <= target_position; ++src_pos)
                    {
                        const int32_t block_id = block_id_for(seq_index, src_pos);
                        const int32_t token_offset = src_pos % metadata.block_size_tokens;
                        const float* value_base = v_block_ptr(block_id)
                            + static_cast<size_t>(token_offset) * static_cast<size_t>(kv_hidden_size(config_));
                        const size_t v_index =
                            static_cast<size_t>(kv_head * config_.head_dim + dim);
                        value += (scores[static_cast<size_t>(src_pos)] / score_sum) * value_base[v_index];
                    }
                    const size_t out_index =
                        q_row_offset + static_cast<size_t>(q_head * config_.head_dim + dim);
                    out_ptr[out_index] = value;
                }
            }
        }
        return;
    }

    std::vector<float> scores(static_cast<size_t>(rows), -std::numeric_limits<float>::infinity());

    for (int64_t row = 0; row < rows; ++row)
    {
        const size_t q_row_offset = static_cast<size_t>(row) * static_cast<size_t>(config_.hidden_size);
        const int32_t target_position = positions_ptr[row];
        for (int32_t q_head = 0; q_head < config_.num_attention_heads; ++q_head)
        {
            const int32_t kv_head = q_head / group_size;
            float max_score = -std::numeric_limits<float>::infinity();

            for (int64_t src = 0; src < rows; ++src)
            {
                if (src > row || positions_ptr[src] > target_position)
                {
                    scores[static_cast<size_t>(src)] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                const size_t k_row_offset =
                    static_cast<size_t>(src) * static_cast<size_t>(kv_hidden_size(config_));
                float score = 0.0f;
                for (int32_t dim = 0; dim < config_.head_dim; ++dim)
                {
                    const size_t q_index =
                        q_row_offset + static_cast<size_t>(q_head * config_.head_dim + dim);
                    const size_t k_index =
                        k_row_offset + static_cast<size_t>(kv_head * config_.head_dim + dim);
                    score += q_ptr[q_index] * k_ptr[k_index];
                }
                score *= scale;
                scores[static_cast<size_t>(src)] = score;
                max_score = std::max(max_score, score);
            }

            float score_sum = 0.0f;
            for (int64_t src = 0; src < rows; ++src)
            {
                if (scores[static_cast<size_t>(src)] == -std::numeric_limits<float>::infinity())
                {
                    continue;
                }
                const float exp_score = std::exp(scores[static_cast<size_t>(src)] - max_score);
                scores[static_cast<size_t>(src)] = exp_score;
                score_sum += exp_score;
            }
            if (score_sum <= 0.0f)
            {
                throw std::runtime_error("LlamaSelfAttention::compute_attention: no causal source tokens.");
            }

            for (int32_t dim = 0; dim < config_.head_dim; ++dim)
            {
                float value = 0.0f;
                for (int64_t src = 0; src < rows; ++src)
                {
                    const float exp_score = scores[static_cast<size_t>(src)];
                    if (exp_score == -std::numeric_limits<float>::infinity())
                    {
                        continue;
                    }
                    const size_t v_row_offset =
                        static_cast<size_t>(src) * static_cast<size_t>(kv_hidden_size(config_));
                    const size_t v_index =
                        v_row_offset + static_cast<size_t>(kv_head * config_.head_dim + dim);
                    value += (exp_score / score_sum) * v_ptr[v_index];
                }
                const size_t out_index =
                    q_row_offset + static_cast<size_t>(q_head * config_.head_dim + dim);
                out_ptr[out_index] = value;
            }
        }
    }
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
    validate_cpu_tensor(gate, "LlamaMLP::apply_activation");
    const size_t count = tensor_numel(gate);
    const float* gate_ptr = static_cast<const float*>(tensor_data(gate));
    const float* up_ptr = static_cast<const float*>(tensor_data(up));
    float* out_ptr = static_cast<float*>(tensor_data(activated));
    for (size_t i = 0; i < count; ++i)
    {
        const float gate_value = gate_ptr[i];
        const float silu = gate_value / (1.0f + std::exp(-gate_value));
        out_ptr[i] = silu * up_ptr[i];
    }
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
    validate_cpu_tensor(src, "LlamaDecoderLayer::copy_tensor");
    const size_t count = tensor_numel(src);
    const float* src_ptr = static_cast<const float*>(tensor_data(src));
    float* dst_ptr = static_cast<float*>(tensor_data(dst));
    for (size_t i = 0; i < count; ++i)
    {
        dst_ptr[i] = src_ptr[i];
    }
}

void LlamaDecoderLayer::add_inplace(const Tensor& residual, const Tensor& update, Tensor& output) const
{
    validate_cpu_tensor(residual, "LlamaDecoderLayer::add_inplace");
    const size_t count = tensor_numel(residual);
    const float* residual_ptr = static_cast<const float*>(tensor_data(residual));
    const float* update_ptr = static_cast<const float*>(tensor_data(update));
    float* output_ptr = static_cast<float*>(tensor_data(output));
    for (size_t i = 0; i < count; ++i)
    {
        output_ptr[i] = residual_ptr[i] + update_ptr[i];
    }
}

} // namespace tiny_llm
