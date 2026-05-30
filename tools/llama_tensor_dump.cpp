#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#define private public
#include "tiny_llm/models/llama_model.h"
#undef private

#include "tiny_llm/core/context.h"
#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/models/hf_safetensors_loader.h"
#include "tiny_llm/models/llama_weight_map.h"

namespace {

int32_t parse_int32(const char* text)
{
    size_t consumed = 0;
    const long value = std::stol(text, &consumed);
    if (consumed != std::string(text).size() || value < INT32_MIN || value > INT32_MAX)
    {
        throw std::runtime_error("invalid int32 token id: " + std::string(text));
    }
    return static_cast<int32_t>(value);
}

std::string layer_name(int32_t layer_id, const std::string& suffix)
{
    std::ostringstream out;
    out << "layer_" << std::setw(2) << std::setfill('0') << layer_id << "_" << suffix;
    return out.str();
}

void write_exact(std::ofstream& out, const void* data, std::streamsize bytes)
{
    out.write(static_cast<const char*>(data), bytes);
    if (!out)
    {
        throw std::runtime_error("failed to write tensor dump.");
    }
}

void dump_tensor(const std::filesystem::path& output_dir,
                 const std::string& name,
                 const tiny_llm::Tensor& tensor)
{
    if (tiny_llm::tensor_dtype(tensor) != tiny_llm::DType::kFloat32)
    {
        throw std::runtime_error("llama_tensor_dump only supports float32 tensors: " + name);
    }
    if (tensor.device().is_cuda())
    {
        throw std::runtime_error("llama_tensor_dump only supports CPU tensors: " + name);
    }

    const tiny_llm::Tensor contiguous = tensor.contiguous();
    const std::filesystem::path path = output_dir / (name + ".bin");
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        throw std::runtime_error("failed to open tensor dump: " + path.string());
    }

    const int32_t rank = static_cast<int32_t>(contiguous.dim());
    write_exact(out, &rank, sizeof(rank));
    for (int32_t i = 0; i < rank; ++i)
    {
        const int64_t dim = contiguous.size(i);
        write_exact(out, &dim, sizeof(dim));
    }
    write_exact(
        out,
        contiguous.data_ptr<float>(),
        static_cast<std::streamsize>(contiguous.numel() * static_cast<int64_t>(sizeof(float))));
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::cerr << "usage: " << argv[0] << " <model_dir> <output_dir> <token0> <token1> [token...]\n";
        return 2;
    }

    try
    {
        const std::string model_dir = argv[1];
        const std::filesystem::path output_dir = argv[2];
        std::vector<int32_t> input_ids_data;
        std::vector<int32_t> positions_data;
        for (int arg = 3; arg < argc; ++arg)
        {
            input_ids_data.push_back(parse_int32(argv[arg]));
            positions_data.push_back(static_cast<int32_t>(positions_data.size()));
        }

        std::filesystem::create_directories(output_dir);

        const std::filesystem::path weight_path = std::filesystem::path(model_dir) / "model.safetensors";
        const tiny_llm::LlamaConfig config = tiny_llm::HFLlamaConfigLoader::load_from_dir(model_dir);
        const tiny_llm::HFSafeTensorLoader loader =
            tiny_llm::HFSafeTensorLoader::from_file(weight_path.string());
        const tiny_llm::WeightMap weight_map = tiny_llm::WeightMap::from_safetensors(loader);

        tiny_llm::LlamaForCausalLM causal_model(config, weight_map);
        const int32_t num_tokens = static_cast<int32_t>(input_ids_data.size());
        causal_model.allocate_buffers(num_tokens);
        tiny_llm::LlamaModel& model = *causal_model.model_;

        torch::Tensor input_ids = torch::from_blob(
            input_ids_data.data(),
            {static_cast<int64_t>(num_tokens)},
            torch::TensorOptions().dtype(torch::kInt32)).clone();
        torch::Tensor positions = torch::from_blob(
            positions_data.data(),
            {static_cast<int64_t>(num_tokens)},
            torch::TensorOptions().dtype(torch::kInt32)).clone();
        torch::Tensor logits = torch::zeros(
            {num_tokens, config.vocab_size},
            torch::TensorOptions().dtype(torch::kFloat32));

        tiny_llm::ExecutionContext ctx(nullptr, nullptr, nullptr);
        tiny_llm::RuntimeContext runtime_ctx(ctx, tiny_llm::ops::PagedAttentionRuntimeMetadata{});
        tiny_llm::LlamaModelBuffers buffers = model.make_batch_buffers(num_tokens);

        model.embed_tokens_->forward(input_ids, buffers.hidden_states);
        dump_tensor(output_dir, "00_embed", buffers.hidden_states);

        for (int32_t layer_id = 0; layer_id < config.num_hidden_layers; ++layer_id)
        {
            const std::shared_ptr<tiny_llm::LlamaDecoderLayer>& layer =
                model.layers_[static_cast<size_t>(layer_id)];

            layer->copy_tensor(buffers.hidden_states, buffers.layer.residual);
            layer->input_layernorm_->forward(buffers.hidden_states, buffers.layer.norm_output, ctx);
            dump_tensor(output_dir, layer_name(layer_id, "input_norm"), buffers.layer.norm_output);

            layer->self_attn_->qkv_proj_->forward(
                buffers.layer.norm_output,
                buffers.layer.attention.qkv,
                ctx);
            layer->self_attn_->split_qkv(
                buffers.layer.attention.qkv,
                buffers.layer.attention.q,
                buffers.layer.attention.k,
                buffers.layer.attention.v);
            dump_tensor(output_dir, layer_name(layer_id, "qkv"), buffers.layer.attention.qkv);
            layer->self_attn_->apply_rope(positions, buffers.layer.attention.q, buffers.layer.attention.k);
            dump_tensor(output_dir, layer_name(layer_id, "q_rope"), buffers.layer.attention.q);
            dump_tensor(output_dir, layer_name(layer_id, "k_rope"), buffers.layer.attention.k);
            dump_tensor(output_dir, layer_name(layer_id, "v"), buffers.layer.attention.v);
            layer->self_attn_->compute_attention(
                positions,
                buffers.layer.attention.q,
                buffers.layer.attention.k,
                buffers.layer.attention.v,
                buffers.layer.attention.attn_output,
                runtime_ctx);
            dump_tensor(output_dir, layer_name(layer_id, "attn_output"), buffers.layer.attention.attn_output);
            layer->self_attn_->o_proj_->forward(
                buffers.layer.attention.attn_output,
                buffers.layer.attention.proj_output,
                ctx);
            dump_tensor(output_dir, layer_name(layer_id, "attn_proj"), buffers.layer.attention.proj_output);
            layer->add_inplace(buffers.layer.residual, buffers.layer.attention.proj_output, buffers.hidden_states);
            dump_tensor(output_dir, layer_name(layer_id, "post_attn_residual"), buffers.hidden_states);

            layer->copy_tensor(buffers.hidden_states, buffers.layer.residual);
            layer->post_attention_layernorm_->forward(buffers.hidden_states, buffers.layer.norm_output, ctx);
            dump_tensor(output_dir, layer_name(layer_id, "post_attn_norm"), buffers.layer.norm_output);
            layer->mlp_->gate_up_proj_->forward(buffers.layer.norm_output, buffers.layer.mlp.gate_up, ctx);
            buffers.layer.mlp.gate.copy_(
                buffers.layer.mlp.gate_up.narrow(
                    1,
                    0,
                    model.config().intermediate_size));
            buffers.layer.mlp.up.copy_(
                buffers.layer.mlp.gate_up.narrow(
                    1,
                    model.config().intermediate_size,
                    model.config().intermediate_size));
            layer->mlp_->apply_activation(buffers.layer.mlp.gate, buffers.layer.mlp.up, buffers.layer.mlp.activated);
            layer->mlp_->down_proj_->forward(buffers.layer.mlp.activated, buffers.layer.mlp.down, ctx);
            dump_tensor(output_dir, layer_name(layer_id, "mlp_gate"), buffers.layer.mlp.gate);
            dump_tensor(output_dir, layer_name(layer_id, "mlp_up"), buffers.layer.mlp.up);
            dump_tensor(output_dir, layer_name(layer_id, "mlp_activated"), buffers.layer.mlp.activated);
            dump_tensor(output_dir, layer_name(layer_id, "mlp_down"), buffers.layer.mlp.down);
            layer->add_inplace(buffers.layer.residual, buffers.layer.mlp.down, buffers.hidden_states);
            dump_tensor(output_dir, layer_name(layer_id, "output"), buffers.hidden_states);
        }

        model.final_norm_->forward(buffers.hidden_states, buffers.norm_output, ctx);
        dump_tensor(output_dir, "final_norm", buffers.norm_output);
        causal_model.lm_head_->forward(buffers.norm_output, logits, ctx);
        dump_tensor(output_dir, "logits", logits);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "llama_tensor_dump failed: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
