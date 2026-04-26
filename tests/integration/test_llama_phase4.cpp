#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define private public
#include "tiny_llm/models/llama_model.h"
#undef private

#include "tiny_llm/core/context.h"
#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/models/hf_safetensors_loader.h"
#include "tiny_llm/models/llama_weight_map.h"

namespace {

bool has_non_zero(const tiny_llm::Tensor& tensor)
{
    const float* ptr = tensor.data_ptr<float>();
    for (size_t i = 0; i < tiny_llm::tensor_numel(tensor); ++i)
    {
        if (std::fabs(ptr[i]) > 1e-7f)
        {
            return true;
        }
    }
    return false;
}

void assert_row_equals_embedding(const tiny_llm::Tensor& row_tensor,
                                 const tiny_llm::Tensor& embedding,
                                 int32_t token_id)
{
    const float* row_ptr = row_tensor.data_ptr<float>();
    const float* emb_ptr = embedding.data_ptr<float>();
    const int64_t stride0 = embedding.stride(0);
    const int64_t stride1 = embedding.stride(1);
    for (int64_t col = 0; col < row_tensor.size(1); ++col)
    {
        const float expected = emb_ptr[
            static_cast<size_t>(token_id) * static_cast<size_t>(stride0)
            + static_cast<size_t>(col) * static_cast<size_t>(stride1)];
        assert(std::fabs(row_ptr[static_cast<size_t>(col)] - expected) <= 1e-6f);
    }
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

    tiny_llm::LlamaModel model(config, weight_map);
    model.allocate_buffers(2);

    const void* hidden_ptr_before = tiny_llm::tensor_data(model.buffers_.hidden_states);
    const void* residual_ptr_before = tiny_llm::tensor_data(model.buffers_.layer.residual);
    const void* qkv_ptr_before = tiny_llm::tensor_data(model.buffers_.layer.attention.qkv);

    int32_t input_ids_data[] = {1, 7};
    int32_t positions_data[] = {0, 1};
    std::vector<float> logits_data(static_cast<size_t>(2 * config.vocab_size), 0.0f);

    const tiny_llm::Tensor input_ids =
        tiny_llm::make_tensor_from_blob(input_ids_data, {2}, tiny_llm::DType::kInt32);
    const tiny_llm::Tensor positions =
        tiny_llm::make_tensor_from_blob(positions_data, {2}, tiny_llm::DType::kInt32);
    tiny_llm::Tensor logits =
        tiny_llm::make_tensor_from_blob(logits_data.data(), {2, config.vocab_size}, tiny_llm::DType::kFloat32);

    tiny_llm::ExecutionContext ctx(nullptr, nullptr, nullptr);

    tiny_llm::Tensor embedding_out = model.make_batch_view_2d(model.buffers_.hidden_states, 2, config.hidden_size);
    model.lookup_embedding(input_ids, embedding_out);
    assert_row_equals_embedding(
        tiny_llm::make_tensor_from_blob(embedding_out.data_ptr<float>(), {1, config.hidden_size}, tiny_llm::DType::kFloat32),
        model.embed_tokens_,
        input_ids_data[0]);
    assert_row_equals_embedding(
        tiny_llm::make_tensor_from_blob(
            embedding_out.data_ptr<float>() + config.hidden_size,
            {1, config.hidden_size},
            tiny_llm::DType::kFloat32),
        model.embed_tokens_,
        input_ids_data[1]);

    bool caught = false;
    try
    {
        int32_t bad_ids_data[] = {config.vocab_size};
        const tiny_llm::Tensor bad_ids =
            tiny_llm::make_tensor_from_blob(bad_ids_data, {1}, tiny_llm::DType::kInt32);
        tiny_llm::Tensor bad_out =
            tiny_llm::make_tensor_from_blob(model.buffers_.hidden_states.data_ptr<float>(), {1, config.hidden_size}, tiny_llm::DType::kFloat32);
        model.lookup_embedding(bad_ids, bad_out);
    }
    catch (const std::runtime_error&)
    {
        caught = true;
    }
    assert(caught);

    model.forward_step(input_ids, positions, logits, ctx);
    assert(logits.size(0) == 2);
    assert(logits.size(1) == config.vocab_size);
    assert(has_non_zero(logits));
    const float* logits_ptr = logits.data_ptr<float>();
    bool rows_differ = false;
    for (int32_t col = 0; col < config.vocab_size; ++col)
    {
        if (std::fabs(logits_ptr[static_cast<size_t>(col)]
                      - logits_ptr[static_cast<size_t>(config.vocab_size + col)]) > 1e-6f)
        {
            rows_differ = true;
            break;
        }
    }
    assert(rows_differ);

    std::fill(logits_data.begin(), logits_data.end(), 0.0f);
    model.forward_step(input_ids, positions, logits, ctx);
    assert(tiny_llm::tensor_data(model.buffers_.hidden_states) == hidden_ptr_before);
    assert(tiny_llm::tensor_data(model.buffers_.layer.residual) == residual_ptr_before);
    assert(tiny_llm::tensor_data(model.buffers_.layer.attention.qkv) == qkv_ptr_before);

    std::cout << "[test_llama_phase4] top-level model checks passed\n";
    return 0;
}
