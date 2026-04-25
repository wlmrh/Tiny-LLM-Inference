#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "tiny_llm/core/context.h"
#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/models/hf_safetensors_loader.h"
#include "tiny_llm/models/llama_decoder_layer.h"
#include "tiny_llm/models/llama_weight_map.h"

namespace {

std::vector<float> make_hidden_data(int rows, int cols)
{
    std::vector<float> out(static_cast<size_t>(rows * cols), 0.0f);
    for (size_t i = 0; i < out.size(); ++i)
    {
        out[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
    }
    return out;
}

tiny_llm::LlamaDecoderLayerBuffers make_buffers(const tiny_llm::LlamaConfig& config,
                                                int rows,
                                                std::vector<float>& residual,
                                                std::vector<float>& norm,
                                                std::vector<float>& qkv,
                                                std::vector<float>& q,
                                                std::vector<float>& k,
                                                std::vector<float>& v,
                                                std::vector<float>& attn_input,
                                                std::vector<float>& attn_output,
                                                std::vector<float>& attn_proj,
                                                std::vector<float>& gate,
                                                std::vector<float>& up,
                                                std::vector<float>& activated,
                                                std::vector<float>& down)
{
    residual.assign(static_cast<size_t>(rows * config.hidden_size), 0.0f);
    norm.assign(static_cast<size_t>(rows * config.hidden_size), 0.0f);
    qkv.assign(static_cast<size_t>(rows * config.hidden_size * 3), 0.0f);
    q.assign(static_cast<size_t>(rows * config.hidden_size), 0.0f);
    k.assign(static_cast<size_t>(rows * config.hidden_size), 0.0f);
    v.assign(static_cast<size_t>(rows * config.hidden_size), 0.0f);
    attn_input.assign(static_cast<size_t>(rows * config.hidden_size), 0.0f);
    attn_output.assign(static_cast<size_t>(rows * config.hidden_size), 0.0f);
    attn_proj.assign(static_cast<size_t>(rows * config.hidden_size), 0.0f);
    gate.assign(static_cast<size_t>(rows * config.intermediate_size), 0.0f);
    up.assign(static_cast<size_t>(rows * config.intermediate_size), 0.0f);
    activated.assign(static_cast<size_t>(rows * config.intermediate_size), 0.0f);
    down.assign(static_cast<size_t>(rows * config.hidden_size), 0.0f);

    tiny_llm::LlamaDecoderLayerBuffers buffers;
    buffers.residual = tiny_llm::make_tensor_from_blob(residual.data(), {rows, config.hidden_size}, tiny_llm::DType::kFloat32);
    buffers.norm_output = tiny_llm::make_tensor_from_blob(norm.data(), {rows, config.hidden_size}, tiny_llm::DType::kFloat32);
    buffers.attention.qkv = tiny_llm::make_tensor_from_blob(qkv.data(), {rows, config.hidden_size * 3}, tiny_llm::DType::kFloat32);
    buffers.attention.q = tiny_llm::make_tensor_from_blob(q.data(), {rows, config.hidden_size}, tiny_llm::DType::kFloat32);
    buffers.attention.k = tiny_llm::make_tensor_from_blob(k.data(), {rows, config.hidden_size}, tiny_llm::DType::kFloat32);
    buffers.attention.v = tiny_llm::make_tensor_from_blob(v.data(), {rows, config.hidden_size}, tiny_llm::DType::kFloat32);
    buffers.attention.attn_input = tiny_llm::make_tensor_from_blob(attn_input.data(), {rows, config.hidden_size}, tiny_llm::DType::kFloat32);
    buffers.attention.attn_output = tiny_llm::make_tensor_from_blob(attn_output.data(), {rows, config.hidden_size}, tiny_llm::DType::kFloat32);
    buffers.attention.proj_output = tiny_llm::make_tensor_from_blob(attn_proj.data(), {rows, config.hidden_size}, tiny_llm::DType::kFloat32);
    buffers.mlp.gate = tiny_llm::make_tensor_from_blob(gate.data(), {rows, config.intermediate_size}, tiny_llm::DType::kFloat32);
    buffers.mlp.up = tiny_llm::make_tensor_from_blob(up.data(), {rows, config.intermediate_size}, tiny_llm::DType::kFloat32);
    buffers.mlp.activated = tiny_llm::make_tensor_from_blob(activated.data(), {rows, config.intermediate_size}, tiny_llm::DType::kFloat32);
    buffers.mlp.down = tiny_llm::make_tensor_from_blob(down.data(), {rows, config.hidden_size}, tiny_llm::DType::kFloat32);
    return buffers;
}

bool differs(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    assert(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        if (std::fabs(lhs[i] - rhs[i]) > 1e-6f)
        {
            return true;
        }
    }
    return false;
}

} // namespace

int main()
{
    const std::string model_dir = "/Users/tangqi/weights";
    assert(std::filesystem::exists(model_dir));

    const tiny_llm::LlamaConfig config = tiny_llm::HFLlamaConfigLoader::load_from_dir(model_dir);
    const tiny_llm::HFSafeTensorLoader loader =
        tiny_llm::HFSafeTensorLoader::from_file(model_dir + "/model.safetensors");
    const tiny_llm::WeightMap weight_map = tiny_llm::WeightMap::from_safetensors(loader);
    tiny_llm::ExecutionContext ctx(nullptr, nullptr, nullptr);

    std::vector<float> hidden_data = make_hidden_data(2, config.hidden_size);
    std::vector<float> hidden_layer0_input = hidden_data;
    std::vector<float> hidden_layer1_input = hidden_data;
    tiny_llm::Tensor hidden0 =
        tiny_llm::make_tensor_from_blob(hidden_layer0_input.data(), {2, config.hidden_size}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor hidden1 =
        tiny_llm::make_tensor_from_blob(hidden_layer1_input.data(), {2, config.hidden_size}, tiny_llm::DType::kFloat32);

    int32_t positions_data[] = {0, 1};
    const tiny_llm::Tensor positions =
        tiny_llm::make_tensor_from_blob(positions_data, {2}, tiny_llm::DType::kInt32);

    std::vector<float> residual0, norm0, qkv0, q0, k0, v0, attn_in0, attn_out0, attn_proj0, gate0, up0, act0, down0;
    std::vector<float> residual1, norm1, qkv1, q1, k1, v1, attn_in1, attn_out1, attn_proj1, gate1, up1, act1, down1;
    tiny_llm::LlamaDecoderLayerBuffers buffers0 = make_buffers(
        config, 2, residual0, norm0, qkv0, q0, k0, v0, attn_in0, attn_out0, attn_proj0, gate0, up0, act0, down0);
    tiny_llm::LlamaDecoderLayerBuffers buffers1 = make_buffers(
        config, 2, residual1, norm1, qkv1, q1, k1, v1, attn_in1, attn_out1, attn_proj1, gate1, up1, act1, down1);

    tiny_llm::LlamaDecoderLayer layer0(config);
    tiny_llm::LlamaDecoderLayer layer1(config);
    layer0.load_weights(weight_map, 0);
    layer1.load_weights(weight_map, 1);

    layer0.forward(hidden0, positions, buffers0, ctx);
    layer1.forward(hidden1, positions, buffers1, ctx);

    assert(differs(hidden_layer0_input, hidden_data));
    assert(differs(hidden_layer1_input, hidden_data));
    assert(differs(hidden_layer0_input, hidden_layer1_input));

    std::cout << "[test_llama_phase3] decoder layer checks passed\n";
    return 0;
}
