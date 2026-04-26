#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/models/hf_safetensors_loader.h"
#include "tiny_llm/models/llama_weight_map.h"

namespace {

std::string shape_to_csv(const std::vector<int64_t>& shape)
{
    if (shape.empty())
    {
        return "scalar";
    }

    std::string out;
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
        {
            out.push_back(',');
        }
        out += std::to_string(shape[i]);
    }
    return out;
}

std::string first_values_to_csv(const float* data, int64_t numel, int32_t count)
{
    if (data == nullptr || numel <= 0 || count <= 0)
    {
        return "";
    }

    const int64_t limit = std::min<int64_t>(numel, static_cast<int64_t>(count));
    std::ostringstream out;
    out << std::setprecision(10);
    for (int64_t i = 0; i < limit; ++i)
    {
        if (i > 0)
        {
            out << ",";
        }
        out << data[i];
    }
    return out.str();
}

void print_tensor_digest(const tiny_llm::HFSafeTensorLoader& loader, const std::string& key)
{
    if (!loader.has_tensor(key))
    {
        throw std::runtime_error("hf_safetensor_dump: missing tensor key: " + key);
    }

    tiny_llm::Tensor tensor = loader.tensor(key).contiguous();
    if (tiny_llm::tensor_dtype(tensor) != tiny_llm::DType::kFloat32)
    {
        throw std::runtime_error("hf_safetensor_dump: only F32 tensor is supported for digest output.");
    }

    const std::vector<int64_t> shape = tiny_llm::tensor_shape(tensor);
    const int64_t numel = static_cast<int64_t>(tiny_llm::tensor_numel(tensor));
    const float* data = tensor.data_ptr<float>();

    double sum = 0.0;
    double l2 = 0.0;
    for (int64_t i = 0; i < numel; ++i)
    {
        const double value = static_cast<double>(data[i]);
        sum += value;
        l2 += value * value;
    }

    std::cout << key << "\t"
              << shape_to_csv(shape) << "\t"
              << numel << "\t"
              << std::setprecision(16) << sum << "\t"
              << std::setprecision(16) << l2 << "\t"
              << first_values_to_csv(data, numel, 8) << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    const std::string model_dir = (argc > 1) ? argv[1] : "/Users/tangqi/models/smollm2-135M";
    const std::string weight_file = (argc > 2) ? argv[2] : "model.safetensors";

    try
    {
        const tiny_llm::LlamaConfig config = tiny_llm::HFLlamaConfigLoader::load_from_dir(model_dir);
        const std::string weight_path = model_dir + "/" + weight_file;
        const tiny_llm::HFSafeTensorLoader loader = tiny_llm::HFSafeTensorLoader::from_file(weight_path);

        // Validate the complete minimal llama key map before exporting tensor digests.
        (void)tiny_llm::load_llama_weights(loader, config);

        std::vector<std::string> keys = {
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.norm.weight",
        };
        keys.push_back(loader.has_tensor("lm_head.weight") ? "lm_head.weight" : "model.embed_tokens.weight");

        if (config.num_hidden_layers > 1)
        {
            keys.push_back("model.layers.1.self_attn.q_proj.weight");
        }

        for (const std::string& key : keys)
        {
            print_tensor_digest(loader, key);
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
