#include "tiny_llm/models/llama_model.h"

#include "tiny_llm/runtime/profiling.h"

#include <limits>
#include <memory>
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

Tensor make_owned_tensor(const std::vector<int64_t>& shape,
                         DType dtype,
                         const c10::Device& device)
{
    return torch::empty(shape, torch::TensorOptions().dtype(to_torch_scalar_type(dtype)).device(device));
}

int32_t kv_hidden_size(const LlamaConfig& config)
{
    return config.num_key_value_heads * config.head_dim;
}

} // namespace

LlamaModel::LlamaModel(LlamaConfig config, const WeightMap& weight_map)
    : config_(std::move(config))
{
    validate_weight_shapes(weight_map);
    embed_tokens_ = register_module(
        "embed_tokens",
        std::make_shared<modules::Embedding>(config_.vocab_size, config_.hidden_size));
    embed_tokens_->bind_weight(weight_map.get_tensor_view("model.embed_tokens.weight"));
    final_norm_ = register_module(
        "final_norm",
        std::make_shared<modules::RMSNorm>(config_.hidden_size, config_.rms_norm_eps));
    final_norm_->bind_weights(weight_map.get_tensor_view("model.norm.weight"));

    layers_.reserve(static_cast<size_t>(config_.num_hidden_layers));
    for (int32_t layer_id = 0; layer_id < config_.num_hidden_layers; ++layer_id)
    {
        auto layer = register_module(
            "layer_" + std::to_string(layer_id),
            std::make_shared<LlamaDecoderLayer>(config_));
        layer->load_weights(weight_map, layer_id);
        layers_.push_back(std::move(layer));
    }
}

void LlamaModel::allocate_buffers(int max_batch_size)
{
    allocate_buffers(max_batch_size, ParallelConfig::cpu());
}

void LlamaModel::allocate_buffers(int max_batch_size, const ParallelConfig& parallel_config)
{
    if (max_batch_size <= 0)
    {
        throw std::runtime_error("LlamaModel::allocate_buffers: max_batch_size must be positive.");
    }
    parallel_config.validate();
    const bool same_device = buffer_parallel_config_ == parallel_config;
    if (same_device && max_batch_size <= allocated_max_batch_size_)
    {
        return;
    }

    allocated_max_batch_size_ = max_batch_size;
    buffer_parallel_config_ = parallel_config;
    const c10::Device device = parallel_config.torch_device();

    buffers_.hidden_states = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32, device);
    buffers_.residual = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32, device);
    buffers_.norm_output = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32, device);
    buffers_.layer.residual = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32, device);
    buffers_.layer.norm_output = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32, device);
    buffers_.layer.attention.qkv =
        make_owned_tensor({max_batch_size, config_.hidden_size + 2 * kv_hidden_size(config_)}, DType::kFloat32, device);
    buffers_.layer.attention.q = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32, device);
    buffers_.layer.attention.k = make_owned_tensor({max_batch_size, kv_hidden_size(config_)}, DType::kFloat32, device);
    buffers_.layer.attention.v = make_owned_tensor({max_batch_size, kv_hidden_size(config_)}, DType::kFloat32, device);
    buffers_.layer.attention.attn_input = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32, device);
    buffers_.layer.attention.attn_output = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32, device);
    buffers_.layer.attention.proj_output = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32, device);
    buffers_.layer.mlp.gate_up = make_owned_tensor({max_batch_size, 2 * config_.intermediate_size}, DType::kFloat32, device);
    buffers_.layer.mlp.gate = make_owned_tensor({max_batch_size, config_.intermediate_size}, DType::kFloat32, device);
    buffers_.layer.mlp.up = make_owned_tensor({max_batch_size, config_.intermediate_size}, DType::kFloat32, device);
    buffers_.layer.mlp.activated = make_owned_tensor({max_batch_size, config_.intermediate_size}, DType::kFloat32, device);
    buffers_.layer.mlp.down = make_owned_tensor({max_batch_size, config_.hidden_size}, DType::kFloat32, device);
}

Tensor LlamaModel::forward_hidden(const PreparedInputs& inputs, RuntimeContext& ctx)
{
    validate_forward_inputs(inputs.input_ids, inputs.positions);

    const int batch_size = checked_positive_dim(inputs.input_ids.size(0), "batch_size");
    if (allocated_max_batch_size_ <= 0)
    {
        throw std::runtime_error("LlamaModel::forward_hidden: buffers are not allocated.");
    }
    if (batch_size > allocated_max_batch_size_)
    {
        throw std::runtime_error("LlamaModel::forward_hidden: batch size exceeds allocated buffers.");
    }

    LlamaModelBuffers batch_buffers = make_batch_buffers(batch_size);
    {
        ScopedRuntimeProfile profile(ctx, &RuntimeProfilingStats::embedding_ms);
        embed_tokens_->forward(inputs.input_ids, batch_buffers.hidden_states);
    }

    for (const std::shared_ptr<LlamaDecoderLayer>& layer : layers_)
    {
        layer->forward(batch_buffers.hidden_states, inputs.positions, batch_buffers.layer, ctx);
    }

    {
        ScopedRuntimeProfile profile(ctx, &RuntimeProfilingStats::norm_ms);
        final_norm_->forward(batch_buffers.hidden_states, batch_buffers.norm_output, ctx.execution());
    }
    return batch_buffers.norm_output;
}

void LlamaModel::validate_forward_inputs(const Tensor& input_ids,
                                         const Tensor& positions) const
{
    if (tensor_dtype(input_ids) != DType::kInt32 || tensor_dtype(positions) != DType::kInt32)
    {
        throw std::runtime_error("LlamaModel::forward_hidden: input_ids and positions must be int32.");
    }

    if (input_ids.dim() != 1 || positions.dim() != 1)
    {
        throw std::runtime_error("LlamaModel::forward_hidden: input_ids and positions must be rank-1.");
    }
    if (!input_ids.sizes().equals(positions.sizes()))
    {
        throw std::runtime_error("LlamaModel::forward_hidden: input_ids and positions must have same shape.");
    }
    if (tensor_data(input_ids) == nullptr || tensor_data(positions) == nullptr)
    {
        throw std::runtime_error("LlamaModel::forward_hidden: input pointers must be non-null.");
    }
}

void LlamaModel::validate_weight_shapes(const WeightMap& weight_map) const
{
    const Tensor& embed_tokens = weight_map.get_tensor_view("model.embed_tokens.weight");
    if (embed_tokens.dim() != 2)
    {
        throw std::runtime_error("LlamaModel: model.embed_tokens.weight must be rank-2.");
    }
    if (!((embed_tokens.size(0) == config_.vocab_size && embed_tokens.size(1) == config_.hidden_size)
          || (embed_tokens.size(0) == config_.hidden_size && embed_tokens.size(1) == config_.vocab_size)))
    {
        throw std::runtime_error("LlamaModel: unsupported embed_tokens shape.");
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
    out.layer.mlp.gate_up = make_batch_view_2d(buffers_.layer.mlp.gate_up, batch_size, 2 * config_.intermediate_size);
    out.layer.mlp.gate = make_batch_view_2d(buffers_.layer.mlp.gate, batch_size, config_.intermediate_size);
    out.layer.mlp.up = make_batch_view_2d(buffers_.layer.mlp.up, batch_size, config_.intermediate_size);
    out.layer.mlp.activated = make_batch_view_2d(buffers_.layer.mlp.activated, batch_size, config_.intermediate_size);
    out.layer.mlp.down = make_batch_view_2d(buffers_.layer.mlp.down, batch_size, config_.hidden_size);
    return out;
}

Tensor LlamaModel::make_batch_view_2d(const Tensor& backing, int batch_size, int width) const
{
    return torch::from_blob(
        tensor_data(backing),
        {batch_size, width},
        torch::TensorOptions().dtype(torch::kFloat32).device(backing.device()));
}

LlamaForCausalLM::LlamaForCausalLM(LlamaConfig config, WeightMap weight_map)
    : config_(std::move(config))
{
    model_ = register_module(
        "model",
        std::make_shared<LlamaModel>(config_, weight_map));
    lm_head_ = register_module(
        "lm_head",
        std::make_shared<modules::Linear>(config_.hidden_size, config_.vocab_size));
    bind_lm_head(weight_map);
}

void LlamaForCausalLM::allocate_buffers(int max_batch_size)
{
    model_->allocate_buffers(max_batch_size);
}

void LlamaForCausalLM::allocate_buffers(int max_batch_size, const ParallelConfig& parallel_config)
{
    model_->allocate_buffers(max_batch_size, parallel_config);
}

Tensor LlamaForCausalLM::forward(const PreparedInputs& inputs, RuntimeContext& ctx)
{
    Tensor hidden_states = model_->forward_hidden(inputs, ctx);
    if (!inputs.sample_row_offsets.empty()
        && static_cast<int64_t>(inputs.sample_row_offsets.size()) < hidden_states.size(0))
    {
        std::vector<int64_t> row_indices;
        row_indices.reserve(inputs.sample_row_offsets.size());
        for (int32_t row : inputs.sample_row_offsets)
        {
            row_indices.push_back(static_cast<int64_t>(row));
        }
        Tensor row_tensor = torch::tensor(
            row_indices,
            torch::TensorOptions().dtype(torch::kInt64).device(hidden_states.device()));
        hidden_states = hidden_states.index_select(0, row_tensor);
    }
    return compute_logits(hidden_states, ctx);
}

Tensor LlamaForCausalLM::compute_logits(const Tensor& hidden_states, RuntimeContext& ctx) const
{
    ScopedRuntimeProfile profile(ctx, &RuntimeProfilingStats::lm_head_ms);
    return lm_head_->forward(hidden_states, ctx.execution());
}

void LlamaForCausalLM::bind_lm_head(const WeightMap& weight_map)
{
    const std::string lm_head_key = weight_map.contains("lm_head.weight")
        ? std::string("lm_head.weight")
        : std::string("model.embed_tokens.weight");
    lm_head_->bind_weight(weight_map.get_tensor_view(lm_head_key), modules::WeightLayout::kOutIn);
}

} // namespace tiny_llm
