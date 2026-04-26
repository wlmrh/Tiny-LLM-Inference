#include "tiny_llm/models/llama_model.h"

#include "tiny_llm/core/context.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace tiny_llm {

namespace {

int checked_positive_dim(int64_t dim, const char* name)
{
    if (dim <= 0)
    {
        throw std::runtime_error(std::string("LlamaModel::forward_step: ") + name + " must be positive.");
    }
    if (dim > std::numeric_limits<int>::max())
    {
        throw std::runtime_error(std::string("LlamaModel::forward_step: ") + name + " is too large.");
    }
    return static_cast<int>(dim);
}

Tensor make_owned_tensor(const std::vector<int64_t>& shape, DType dtype)
{
    return torch::empty(shape, torch::TensorOptions().dtype(to_torch_scalar_type(dtype)).device(c10::kCPU));
}

int32_t kv_hidden_size(const LlamaConfig& config)
{
    return config.num_key_value_heads * config.head_dim;
}

} // namespace

LlamaModel::LlamaModel(LlamaConfig config, WeightMap weight_map)
    : config_(std::move(config)),
      weight_map_(std::move(weight_map)),
      final_norm_(config_.hidden_size, config_.rms_norm_eps),
      lm_head_(config_.hidden_size, config_.vocab_size)
{
    validate_weight_shapes();
    layers_.reserve(static_cast<size_t>(config_.num_hidden_layers));
    for (int32_t layer_id = 0; layer_id < config_.num_hidden_layers; ++layer_id)
    {
        layers_.emplace_back(config_);
        layers_.back().load_weights(weight_map_, layer_id);
    }
    bind_top_level_weights();
}

void LlamaModel::allocate_buffers(int max_batch_size)
{
    if (max_batch_size <= 0)
    {
        throw std::runtime_error("LlamaModel::allocate_buffers: max_batch_size must be positive.");
    }
    if (max_batch_size <= allocated_max_batch_size_)
    {
        return;
    }

    allocated_max_batch_size_ = max_batch_size;

    buffers_.hidden_states = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32);
    buffers_.residual = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32);
    buffers_.norm_output = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32);
    buffers_.layer.residual = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32);
    buffers_.layer.norm_output = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32);
    buffers_.layer.attention.qkv =
        make_owned_tensor({max_batch_size, config_.hidden_size + 2 * kv_hidden_size(config_)}, DType::kFloat32);
    buffers_.layer.attention.q = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32);
    buffers_.layer.attention.k = make_owned_tensor({max_batch_size, kv_hidden_size(config_)}, DType::kFloat32);
    buffers_.layer.attention.v = make_owned_tensor({max_batch_size, kv_hidden_size(config_)}, DType::kFloat32);
    buffers_.layer.attention.attn_input = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32);
    buffers_.layer.attention.attn_output = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32);
    buffers_.layer.attention.proj_output = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32);
    buffers_.layer.mlp.gate = make_owned_tensor({max_batch_size, config_.intermediate_size}, DType::kFloat32);
    buffers_.layer.mlp.up = make_owned_tensor({max_batch_size, config_.intermediate_size}, DType::kFloat32);
    buffers_.layer.mlp.activated = make_owned_tensor({max_batch_size, config_.intermediate_size}, DType::kFloat32);
    buffers_.layer.mlp.down = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32);
}

void LlamaModel::forward_step(const Tensor& input_ids,
                              const Tensor& positions,
                              Tensor& logits,
                              ExecutionContext& ctx)
{
    validate_forward_inputs(input_ids, positions, logits);

    const int batch_size = checked_positive_dim(input_ids.size(0), "batch_size");
    if (allocated_max_batch_size_ <= 0)
    {
        throw std::runtime_error("LlamaModel::forward_step: buffers are not allocated.");
    }
    if (batch_size > allocated_max_batch_size_)
    {
        throw std::runtime_error("LlamaModel::forward_step: batch size exceeds allocated buffers.");
    }

    LlamaModelBuffers batch_buffers = make_batch_buffers(batch_size);
    lookup_embedding(input_ids, batch_buffers.hidden_states);

    for (const LlamaDecoderLayer& layer : layers_)
    {
        layer.forward(batch_buffers.hidden_states, positions, batch_buffers.layer, ctx);
    }

    final_norm_.forward(batch_buffers.hidden_states, batch_buffers.norm_output, ctx);
    lm_head_.forward(batch_buffers.norm_output, logits, ctx);
}

void LlamaModel::validate_forward_inputs(const Tensor& input_ids,
                                         const Tensor& positions,
                                         const Tensor& logits) const
{
    if (tensor_dtype(input_ids) != DType::kInt32 || tensor_dtype(positions) != DType::kInt32)
    {
        throw std::runtime_error("LlamaModel::forward_step: input_ids and positions must be int32.");
    }
    if (tensor_dtype(logits) != DType::kFloat32)
    {
        throw std::runtime_error("LlamaModel::forward_step: logits must be float32.");
    }

    if (input_ids.dim() != 1 || positions.dim() != 1)
    {
        throw std::runtime_error("LlamaModel::forward_step: input_ids and positions must be rank-1.");
    }
    if (!input_ids.sizes().equals(positions.sizes()))
    {
        throw std::runtime_error("LlamaModel::forward_step: input_ids and positions must have same shape.");
    }
    if (logits.dim() != 2 || logits.size(0) != input_ids.size(0) || logits.size(1) != config_.vocab_size)
    {
        throw std::runtime_error("LlamaModel::forward_step: logits shape must be [B, vocab_size].");
    }
    if (tensor_data(input_ids) == nullptr || tensor_data(positions) == nullptr || tensor_data(logits) == nullptr)
    {
        throw std::runtime_error("LlamaModel::forward_step: input/output pointers must be non-null.");
    }
}

void LlamaModel::validate_weight_shapes()
{
    embed_tokens_ = weight_map_.get_tensor_view("model.embed_tokens.weight");
    if (embed_tokens_.dim() != 2)
    {
        throw std::runtime_error("LlamaModel: model.embed_tokens.weight must be rank-2.");
    }

    if (embed_tokens_.size(0) == config_.vocab_size && embed_tokens_.size(1) == config_.hidden_size)
    {
        embedding_layout_ = EmbeddingLayout::kVocabHidden;
    }
    else if (embed_tokens_.size(0) == config_.hidden_size && embed_tokens_.size(1) == config_.vocab_size)
    {
        embedding_layout_ = EmbeddingLayout::kHiddenVocab;
    }
    else
    {
        throw std::runtime_error("LlamaModel: unsupported embed_tokens shape.");
    }
}

void LlamaModel::bind_top_level_weights()
{
    final_norm_.bind_weights(weight_map_.get_tensor_as<float>("model.norm.weight"));
    const std::string lm_head_key = weight_map_.contains("lm_head.weight")
        ? std::string("lm_head.weight")
        : std::string("model.embed_tokens.weight");
    lm_head_.bind_weight(
        weight_map_.get_tensor_as<float>(lm_head_key),
        config_.vocab_size,
        config_.hidden_size,
        modules::WeightLayout::kOutIn);
}

void LlamaModel::lookup_embedding(const Tensor& ids, Tensor& out) const
{
    if (tensor_dtype(ids) != DType::kInt32 || tensor_dtype(out) != DType::kFloat32)
    {
        throw std::runtime_error("LlamaModel::lookup_embedding: ids/out dtype mismatch.");
    }
    if (ids.dim() != 1 || out.dim() != 2 || out.size(0) != ids.size(0) || out.size(1) != config_.hidden_size)
    {
        throw std::runtime_error("LlamaModel::lookup_embedding: ids/out shape mismatch.");
    }
    if (embed_tokens_.device().is_cuda() || out.device().is_cuda())
    {
        throw std::runtime_error("LlamaModel::lookup_embedding: CUDA path is not implemented.");
    }

    const int32_t* ids_ptr = static_cast<const int32_t*>(tensor_data(ids));
    const float* embed_ptr = static_cast<const float*>(tensor_data(embed_tokens_));
    float* out_ptr = static_cast<float*>(tensor_data(out));

    const int64_t stride0 = embed_tokens_.stride(0);
    const int64_t stride1 = embed_tokens_.stride(1);
    const int64_t rows = ids.size(0);
    for (int64_t row = 0; row < rows; ++row)
    {
        const int32_t token_id = ids_ptr[row];
        if (token_id < 0 || token_id >= config_.vocab_size)
        {
            throw std::runtime_error("LlamaModel::lookup_embedding: token id is out of range.");
        }

        float* out_row = out_ptr + static_cast<size_t>(row) * static_cast<size_t>(config_.hidden_size);
        for (int32_t col = 0; col < config_.hidden_size; ++col)
        {
            if (embedding_layout_ == EmbeddingLayout::kVocabHidden)
            {
                out_row[static_cast<size_t>(col)] = embed_ptr[
                    static_cast<size_t>(token_id) * static_cast<size_t>(stride0)
                    + static_cast<size_t>(col) * static_cast<size_t>(stride1)];
            }
            else
            {
                out_row[static_cast<size_t>(col)] = embed_ptr[
                    static_cast<size_t>(col) * static_cast<size_t>(stride0)
                    + static_cast<size_t>(token_id) * static_cast<size_t>(stride1)];
            }
        }
    }
}

LlamaModelBuffers LlamaModel::make_batch_buffers(int batch_size) const
{
    LlamaModelBuffers out;
    out.hidden_states = make_batch_view_2d(buffers_.hidden_states, batch_size, config_.hidden_size);
    out.residual = make_batch_view_2d(buffers_.residual, batch_size, config_.hidden_size);
    out.norm_output = make_batch_view_2d(buffers_.norm_output, batch_size, config_.hidden_size);
    out.layer.residual = make_batch_view_2d(buffers_.layer.residual, batch_size, config_.hidden_size);
    out.layer.norm_output = make_batch_view_2d(buffers_.layer.norm_output, batch_size, config_.hidden_size);
    out.layer.attention.qkv =
        make_batch_view_2d(buffers_.layer.attention.qkv, batch_size, config_.hidden_size + 2 * kv_hidden_size(config_));
    out.layer.attention.q = make_batch_view_2d(buffers_.layer.attention.q, batch_size, config_.hidden_size);
    out.layer.attention.k = make_batch_view_2d(buffers_.layer.attention.k, batch_size, kv_hidden_size(config_));
    out.layer.attention.v = make_batch_view_2d(buffers_.layer.attention.v, batch_size, kv_hidden_size(config_));
    out.layer.attention.attn_input = make_batch_view_2d(buffers_.layer.attention.attn_input, batch_size, config_.hidden_size);
    out.layer.attention.attn_output = make_batch_view_2d(buffers_.layer.attention.attn_output, batch_size, config_.hidden_size);
    out.layer.attention.proj_output = make_batch_view_2d(buffers_.layer.attention.proj_output, batch_size, config_.hidden_size);
    out.layer.mlp.gate = make_batch_view_2d(buffers_.layer.mlp.gate, batch_size, config_.intermediate_size);
    out.layer.mlp.up = make_batch_view_2d(buffers_.layer.mlp.up, batch_size, config_.intermediate_size);
    out.layer.mlp.activated = make_batch_view_2d(buffers_.layer.mlp.activated, batch_size, config_.intermediate_size);
    out.layer.mlp.down = make_batch_view_2d(buffers_.layer.mlp.down, batch_size, config_.hidden_size);
    return out;
}

Tensor LlamaModel::make_batch_view_2d(const Tensor& backing, int batch_size, int width) const
{
    return make_tensor_from_blob(tensor_data(backing), {batch_size, width}, DType::kFloat32);
}

} // namespace tiny_llm
