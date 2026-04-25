#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>

#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/models/hf_safetensors_loader.h"
#include "tiny_llm/models/llama_weight_map.h"

int main()
{
    std::string hf_model_dir = "/Users/tangqi/weights";
    assert(std::filesystem::exists(hf_model_dir));

    const tiny_llm::LlamaConfig config =
        tiny_llm::HFLlamaConfigLoader::load_from_dir(hf_model_dir);
    assert(config.model_type == "llama");
    assert(config.hidden_size > 0);
    assert(config.num_hidden_layers > 0);
    assert(config.num_attention_heads > 0);
    assert(config.num_key_value_heads > 0);
    assert(config.vocab_size > 0);
    assert(config.head_dim == config.hidden_size / config.num_attention_heads);

    const tiny_llm::HFSafeTensorLoader loader =
        tiny_llm::HFSafeTensorLoader::from_file(hf_model_dir + "/model.safetensors");
    const tiny_llm::WeightMap weight_map = tiny_llm::WeightMap::from_safetensors(loader);

    assert(weight_map.contains("model.embed_tokens.weight"));
    assert(weight_map.contains("model.layers.0.self_attn.q_proj.weight"));
    assert(weight_map.get_tensor("model.layers.0.self_attn.q_proj.weight") != nullptr);

    const tiny_llm::Tensor& q_proj =
        weight_map.get_tensor_view("model.layers.0.self_attn.q_proj.weight");
    const std::vector<int64_t> q_shape = tiny_llm::tensor_shape(q_proj);
    assert(q_shape.size() == 2);
    assert(q_shape[0] == config.hidden_size);
    assert(q_shape[1] == config.hidden_size);

    (void)tiny_llm::load_llama_weights(weight_map, config);

    std::cout << "[test_llama_phase1] config + weight map checks passed\n";
    return 0;
}
