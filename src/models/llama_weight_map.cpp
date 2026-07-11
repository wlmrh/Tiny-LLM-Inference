#include "tiny_llm/models/llama_weight_map.h"

#include "tiny_llm/models/hf_safetensors_loader.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace tiny_llm {

namespace {

std::string shape_to_string(const std::vector<int64_t>& shape)
{
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        out += std::to_string(shape[i]);
        if (i + 1 < shape.size())
        {
            out += ", ";
        }
    }
    out += "]";
    return out;
}

int32_t kv_hidden_size(const LlamaConfig& config)
{
    return config.num_key_value_heads * config.head_dim;
}

Tensor load_checked_tensor(const WeightMap& weight_map,
                          const std::string& key,
                          const std::vector<int64_t>& expected_shape)
{
    if (!weight_map.contains(key))
    {
        throw std::runtime_error("load_llama_weights: missing tensor key: " + key);
    }

    const Tensor& tensor = weight_map.get_tensor_view(key);
    if (tensor_dtype(tensor) != DType::kFloat32)
    {
        throw std::runtime_error("load_llama_weights: dtype mismatch for key (expect F32): " + key);
    }

    const std::vector<int64_t> loaded_shape = tensor_shape(tensor);
    if (loaded_shape != expected_shape)
    {
        throw std::runtime_error(
            "load_llama_weights: shape mismatch for key " + key
            + ", expected=" + shape_to_string(expected_shape)
            + ", got=" + shape_to_string(loaded_shape));
    }

    return tensor;
}

} // namespace

WeightMap WeightMap::from_safetensors(const HFSafeTensorLoader& loader)
{
    return WeightMap::from_safetensors(loader, ParallelConfig::cpu());
}

WeightMap WeightMap::from_safetensors(const HFSafeTensorLoader& loader,
                                      const ParallelConfig& parallel_config)
{
    WeightMap weight_map;
    parallel_config.validate();
    const c10::Device target_device = parallel_config.torch_device();
    for (const std::string& key : loader.keys())
    {
        Tensor tensor = loader.tensor(key);
        if (tensor.device() != target_device)
        {
            tensor = tensor.to(target_device, /*non_blocking=*/false, /*copy=*/true).contiguous();
        }
        else if (!tensor.is_contiguous())
        {
            tensor = tensor.contiguous();
        }
        weight_map.add_tensor(key, tensor);
    }
    return weight_map;
}

void WeightMap::add_tensor(const std::string& name, const Tensor& tensor)
{
    if (name.empty())
    {
        throw std::runtime_error("WeightMap::add_tensor: tensor name must be non-empty.");
    }
    if (!tensor.defined())
    {
        throw std::runtime_error("WeightMap::add_tensor: tensor must be defined for key: " + name);
    }
    if (tensor_data(tensor) == nullptr)
    {
        throw std::runtime_error("WeightMap::add_tensor: tensor data pointer must be non-null for key: " + name);
    }

    tensor_views_[name] = tensor;
    tensor_ptrs_[name] = tensor_data(tensor_views_.at(name));
}

void WeightMap::add_tensor(const std::string& name,
                           void* data,
                           const std::vector<int64_t>& shape,
                           DType dtype)
{
    add_tensor(name, make_tensor_from_blob(data, shape, dtype));
}

bool WeightMap::contains(const std::string& name) const
{
    return tensor_ptrs_.find(name) != tensor_ptrs_.end();
}

void* WeightMap::get_tensor(const std::string& name) const
{
    const auto it = tensor_ptrs_.find(name);
    if (it == tensor_ptrs_.end())
    {
        throw std::runtime_error("WeightMap::get_tensor: missing tensor key: " + name);
    }
    return it->second;
}

const Tensor& WeightMap::get_tensor_view(const std::string& name) const
{
    const auto it = tensor_views_.find(name);
    if (it == tensor_views_.end())
    {
        throw std::runtime_error("WeightMap::get_tensor_view: missing tensor key: " + name);
    }
    return it->second;
}

std::vector<std::string> WeightMap::keys() const
{
    std::vector<std::string> out;
    out.reserve(tensor_ptrs_.size());
    for (const auto& item : tensor_ptrs_)
    {
        out.push_back(item.first);
    }
    std::sort(out.begin(), out.end());
    return out;
}

LlamaWeights load_llama_weights(const HFSafeTensorLoader& loader,
                                const LlamaConfig& config)
{
    return load_llama_weights(WeightMap::from_safetensors(loader), config);
}

LlamaWeights load_llama_weights(const HFSafeTensorLoader& loader,
                                const LlamaConfig& config,
                                const ParallelConfig& parallel_config)
{
    return load_llama_weights(WeightMap::from_safetensors(loader, parallel_config), config);
}

LlamaWeights load_llama_weights(const WeightMap& weight_map,
                                const LlamaConfig& config)
{
    if (config.num_hidden_layers <= 0 || config.hidden_size <= 0
        || config.intermediate_size <= 0 || config.vocab_size <= 0)
    {
        throw std::runtime_error("load_llama_weights: invalid model config dimensions.");
    }

    LlamaWeights weights;
    weights.embed_tokens = load_checked_tensor(
        weight_map,
        "model.embed_tokens.weight",
        {config.vocab_size, config.hidden_size});

    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int32_t layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx)
    {
        const std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

        LlamaLayerWeights layer;
        layer.input_layernorm = load_checked_tensor(
            weight_map,
            prefix + "input_layernorm.weight",
            {config.hidden_size});
        layer.q_proj = load_checked_tensor(
            weight_map,
            prefix + "self_attn.q_proj.weight",
            {config.hidden_size, config.hidden_size});
        layer.k_proj = load_checked_tensor(
            weight_map,
            prefix + "self_attn.k_proj.weight",
            {kv_hidden_size(config), config.hidden_size});
        layer.v_proj = load_checked_tensor(
            weight_map,
            prefix + "self_attn.v_proj.weight",
            {kv_hidden_size(config), config.hidden_size});
        layer.o_proj = load_checked_tensor(
            weight_map,
            prefix + "self_attn.o_proj.weight",
            {config.hidden_size, config.hidden_size});
        layer.post_attention_layernorm = load_checked_tensor(
            weight_map,
            prefix + "post_attention_layernorm.weight",
            {config.hidden_size});
        layer.gate_proj = load_checked_tensor(
            weight_map,
            prefix + "mlp.gate_proj.weight",
            {config.intermediate_size, config.hidden_size});
        layer.up_proj = load_checked_tensor(
            weight_map,
            prefix + "mlp.up_proj.weight",
            {config.intermediate_size, config.hidden_size});
        layer.down_proj = load_checked_tensor(
            weight_map,
            prefix + "mlp.down_proj.weight",
            {config.hidden_size, config.intermediate_size});

        weights.layers.push_back(std::move(layer));
    }

    weights.norm = load_checked_tensor(weight_map, "model.norm.weight", {config.hidden_size});
    weights.lm_head = load_checked_tensor(
        weight_map,
        weight_map.contains("lm_head.weight") ? "lm_head.weight" : "model.embed_tokens.weight",
        {config.vocab_size, config.hidden_size});

    return weights;
}

} // namespace tiny_llm
