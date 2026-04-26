#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "tiny_llm/core/context.h"
#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/models/hf_safetensors_loader.h"
#include "tiny_llm/models/llama_weight_map.h"
#include "tiny_llm/models/modules/linear.h"
#include "tiny_llm/models/modules/rmsnorm.h"
#include "tiny_llm/operators/matmul.h"
#include "tiny_llm/operators/rmsnorm.h"

namespace {

bool nearly_equal(float lhs, float rhs, float atol = 1e-5f)
{
    return std::fabs(lhs - rhs) <= atol;
}

void assert_tensor_allclose(const tiny_llm::Tensor& lhs,
                            const tiny_llm::Tensor& rhs,
                            float atol = 1e-5f)
{
    assert(lhs.sizes().equals(rhs.sizes()));
    const float* lhs_ptr = lhs.data_ptr<float>();
    const float* rhs_ptr = rhs.data_ptr<float>();
    const size_t count = tiny_llm::tensor_numel(lhs);
    for (size_t i = 0; i < count; ++i)
    {
        assert(nearly_equal(lhs_ptr[i], rhs_ptr[i], atol));
    }
}

void test_rmsnorm_module_matches_op(tiny_llm::ExecutionContext& ctx)
{
    std::vector<float> x_data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 0.5f, 2.5f, -3.0f,
    };
    std::vector<float> w_data = {1.0f, 0.5f, 1.5f, 2.0f};
    std::vector<float> y_module_data(8, 0.0f);
    std::vector<float> y_op_data(8, 0.0f);

    const tiny_llm::Tensor x = tiny_llm::make_tensor_from_blob(x_data.data(), {2, 4}, tiny_llm::DType::kFloat32);
    const tiny_llm::Tensor w = tiny_llm::make_tensor_from_blob(w_data.data(), {4}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor y_module = tiny_llm::make_tensor_from_blob(y_module_data.data(), {2, 4}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor y_op = tiny_llm::make_tensor_from_blob(y_op_data.data(), {2, 4}, tiny_llm::DType::kFloat32);

    tiny_llm::modules::RMSNorm rmsnorm(4, 1e-6f);
    rmsnorm.bind_weights(w_data.data());
    rmsnorm.forward(x, y_module, ctx);
    tiny_llm::ops::rmsnorm(x, w, y_op, ctx, 1e-6f);
    assert_tensor_allclose(y_module, y_op);

    bool caught = false;
    try
    {
        std::vector<float> wrong_output(6, 0.0f);
        tiny_llm::Tensor bad = tiny_llm::make_tensor_from_blob(wrong_output.data(), {2, 3}, tiny_llm::DType::kFloat32);
        rmsnorm.forward(x, bad, ctx);
    }
    catch (const std::runtime_error&)
    {
        caught = true;
    }
    assert(caught);
}

void test_linear_single_weight_matches_gemm(tiny_llm::ExecutionContext& ctx)
{
    std::vector<float> input_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    std::vector<float> weight_data = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    std::vector<float> module_output_data(4, 0.0f);
    std::vector<float> op_output_data(4, 0.0f);

    const tiny_llm::Tensor input = tiny_llm::make_tensor_from_blob(input_data.data(), {2, 3}, tiny_llm::DType::kFloat32);
    const tiny_llm::Tensor weight = tiny_llm::make_tensor_from_blob(weight_data.data(), {3, 2}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor module_output = tiny_llm::make_tensor_from_blob(module_output_data.data(), {2, 2}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor op_output = tiny_llm::make_tensor_from_blob(op_output_data.data(), {2, 2}, tiny_llm::DType::kFloat32);

    tiny_llm::modules::Linear linear(3, 2);
    linear.bind_weight(weight_data.data(), 2, 3);
    linear.forward(input, module_output, ctx);
    tiny_llm::ops::gemm(input, weight, op_output, ctx);

    assert_tensor_allclose(module_output, op_output);
}

void test_linear_stacked_weights_match_separate_gemms(tiny_llm::ExecutionContext& ctx)
{
    std::vector<float> input_data = {
        0.5f, 1.0f, -2.0f,
        1.5f, -0.5f, 3.0f,
    };
    std::vector<float> q_weight_data = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        2.0f, -1.0f,
    };
    std::vector<float> k_weight_data = {
        -1.0f, 1.0f,
        2.0f, 0.5f,
        0.0f, 3.0f,
    };
    std::vector<float> v_weight_data = {
        0.5f, -0.5f,
        1.0f, 2.0f,
        -1.0f, 1.5f,
    };
    std::vector<float> module_output_data(12, 0.0f);
    std::vector<float> expected_output_data(12, 0.0f);
    std::vector<float> q_output_data(4, 0.0f);
    std::vector<float> k_output_data(4, 0.0f);
    std::vector<float> v_output_data(4, 0.0f);

    const tiny_llm::Tensor input = tiny_llm::make_tensor_from_blob(input_data.data(), {2, 3}, tiny_llm::DType::kFloat32);
    const tiny_llm::Tensor q_weight = tiny_llm::make_tensor_from_blob(q_weight_data.data(), {3, 2}, tiny_llm::DType::kFloat32);
    const tiny_llm::Tensor k_weight = tiny_llm::make_tensor_from_blob(k_weight_data.data(), {3, 2}, tiny_llm::DType::kFloat32);
    const tiny_llm::Tensor v_weight = tiny_llm::make_tensor_from_blob(v_weight_data.data(), {3, 2}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor module_output = tiny_llm::make_tensor_from_blob(module_output_data.data(), {2, 6}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor expected_output = tiny_llm::make_tensor_from_blob(expected_output_data.data(), {2, 6}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor q_output = tiny_llm::make_tensor_from_blob(q_output_data.data(), {2, 2}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor k_output = tiny_llm::make_tensor_from_blob(k_output_data.data(), {2, 2}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor v_output = tiny_llm::make_tensor_from_blob(v_output_data.data(), {2, 2}, tiny_llm::DType::kFloat32);

    const tiny_llm::modules::StackedWeightDesc descs[] = {
        {q_weight_data.data(), 2, 3, 0},
        {k_weight_data.data(), 2, 3, 2},
        {v_weight_data.data(), 2, 3, 4},
    };

    tiny_llm::modules::Linear linear(3, 6);
    linear.bind_stacked_weights(descs, 3);
    linear.forward(input, module_output, ctx);

    tiny_llm::ops::gemm(input, q_weight, q_output, ctx);
    tiny_llm::ops::gemm(input, k_weight, k_output, ctx);
    tiny_llm::ops::gemm(input, v_weight, v_output, ctx);

    float* expected_ptr = expected_output.data_ptr<float>();
    const float* q_ptr = q_output.data_ptr<float>();
    const float* k_ptr = k_output.data_ptr<float>();
    const float* v_ptr = v_output.data_ptr<float>();
    for (int row = 0; row < 2; ++row)
    {
        const int row_offset = row * 6;
        expected_ptr[row_offset + 0] = q_ptr[row * 2 + 0];
        expected_ptr[row_offset + 1] = q_ptr[row * 2 + 1];
        expected_ptr[row_offset + 2] = k_ptr[row * 2 + 0];
        expected_ptr[row_offset + 3] = k_ptr[row * 2 + 1];
        expected_ptr[row_offset + 4] = v_ptr[row * 2 + 0];
        expected_ptr[row_offset + 5] = v_ptr[row * 2 + 1];
    }

    assert_tensor_allclose(module_output, expected_output);
}

void test_phase2_smoke_with_weight_map(tiny_llm::ExecutionContext& ctx)
{
    const std::string model_dir = "/Users/tangqi/weights";
    assert(std::filesystem::exists(model_dir));

    const tiny_llm::LlamaConfig config = tiny_llm::HFLlamaConfigLoader::load_from_dir(model_dir);
    const tiny_llm::HFSafeTensorLoader loader =
        tiny_llm::HFSafeTensorLoader::from_file(model_dir + "/model.safetensors");
    const tiny_llm::WeightMap weight_map = tiny_llm::WeightMap::from_safetensors(loader);

    std::vector<float> hidden_data(static_cast<size_t>(config.hidden_size) * 2U, 0.0f);
    for (size_t i = 0; i < hidden_data.size(); ++i)
    {
        hidden_data[i] = static_cast<float>(static_cast<int>(i) - 5) * 0.1f;
    }
    std::vector<float> norm_output_data(hidden_data.size(), 0.0f);
    std::vector<float> qkv_output_data(static_cast<size_t>(2 * config.hidden_size * 3), 0.0f);
    std::vector<float> q_output_data(static_cast<size_t>(2 * config.hidden_size), 0.0f);
    std::vector<float> k_output_data(static_cast<size_t>(2 * config.hidden_size), 0.0f);
    std::vector<float> v_output_data(static_cast<size_t>(2 * config.hidden_size), 0.0f);
    std::vector<float> expected_qkv_data(static_cast<size_t>(2 * config.hidden_size * 3), 0.0f);

    const tiny_llm::Tensor hidden =
        tiny_llm::make_tensor_from_blob(hidden_data.data(), {2, config.hidden_size}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor norm_output =
        tiny_llm::make_tensor_from_blob(norm_output_data.data(), {2, config.hidden_size}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor qkv_output =
        tiny_llm::make_tensor_from_blob(qkv_output_data.data(), {2, config.hidden_size * 3}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor q_output =
        tiny_llm::make_tensor_from_blob(q_output_data.data(), {2, config.hidden_size}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor k_output =
        tiny_llm::make_tensor_from_blob(k_output_data.data(), {2, config.hidden_size}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor v_output =
        tiny_llm::make_tensor_from_blob(v_output_data.data(), {2, config.hidden_size}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor expected_qkv =
        tiny_llm::make_tensor_from_blob(expected_qkv_data.data(), {2, config.hidden_size * 3}, tiny_llm::DType::kFloat32);

    tiny_llm::modules::RMSNorm norm(config.hidden_size, config.rms_norm_eps);
    norm.bind_weights(weight_map.get_tensor_as<float>("model.layers.0.input_layernorm.weight"));
    norm.forward(hidden, norm_output, ctx);

    const tiny_llm::modules::StackedWeightDesc qkv_descs[] = {
        {weight_map.get_tensor_as<float>("model.layers.0.self_attn.q_proj.weight"), config.hidden_size, config.hidden_size, 0},
        {weight_map.get_tensor_as<float>("model.layers.0.self_attn.k_proj.weight"), config.hidden_size, config.hidden_size, config.hidden_size},
        {weight_map.get_tensor_as<float>("model.layers.0.self_attn.v_proj.weight"), config.hidden_size, config.hidden_size, config.hidden_size * 2},
    };

    tiny_llm::modules::Linear qkv_proj(config.hidden_size, config.hidden_size * 3);
    qkv_proj.bind_stacked_weights(qkv_descs, 3);
    qkv_proj.forward(norm_output, qkv_output, ctx);

    const tiny_llm::Tensor q_weight = weight_map.get_tensor_view("model.layers.0.self_attn.q_proj.weight");
    const tiny_llm::Tensor k_weight = weight_map.get_tensor_view("model.layers.0.self_attn.k_proj.weight");
    const tiny_llm::Tensor v_weight = weight_map.get_tensor_view("model.layers.0.self_attn.v_proj.weight");
    tiny_llm::ops::gemm(norm_output, q_weight, q_output, ctx);
    tiny_llm::ops::gemm(norm_output, k_weight, k_output, ctx);
    tiny_llm::ops::gemm(norm_output, v_weight, v_output, ctx);

    float* expected_ptr = expected_qkv.data_ptr<float>();
    const float* q_ptr = q_output.data_ptr<float>();
    const float* k_ptr = k_output.data_ptr<float>();
    const float* v_ptr = v_output.data_ptr<float>();
    for (int row = 0; row < 2; ++row)
    {
        const int row_offset = row * (config.hidden_size * 3);
        const int single_offset = row * config.hidden_size;
        for (int col = 0; col < config.hidden_size; ++col)
        {
            expected_ptr[row_offset + col] = q_ptr[single_offset + col];
            expected_ptr[row_offset + config.hidden_size + col] = k_ptr[single_offset + col];
            expected_ptr[row_offset + config.hidden_size * 2 + col] = v_ptr[single_offset + col];
        }
    }

    assert_tensor_allclose(qkv_output, expected_qkv, 1e-4f);
}

} // namespace

int main()
{
    tiny_llm::ExecutionContext ctx(nullptr, nullptr, nullptr);

    test_rmsnorm_module_matches_op(ctx);
    test_linear_single_weight_matches_gemm(ctx);
    test_linear_stacked_weights_match_separate_gemms(ctx);
    test_phase2_smoke_with_weight_map(ctx);

    std::cout << "[test_llama_phase2] module checks passed\n";
    return 0;
}
