#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "tiny_llm/core/context.h"
#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/models/hf_safetensors_loader.h"
#include "tiny_llm/models/llama_model.h"
#include "tiny_llm/models/llama_weight_map.h"

namespace {

int32_t parse_int32(const char* text, const char* name)
{
    try
    {
        size_t consumed = 0;
        const long value = std::stol(text, &consumed);
        if (consumed != std::string(text).size())
        {
            throw std::runtime_error("trailing characters");
        }
        if (value < INT32_MIN || value > INT32_MAX)
        {
            throw std::runtime_error("out of int32 range");
        }
        return static_cast<int32_t>(value);
    }
    catch (const std::exception& ex)
    {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text + " (" + ex.what() + ")");
    }
}

void write_exact(std::ofstream& out, const void* data, std::streamsize bytes)
{
    out.write(static_cast<const char*>(data), bytes);
    if (!out)
    {
        throw std::runtime_error("failed to write logits output file.");
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::cerr << "usage: " << argv[0] << " <model_dir> <output_bin> <token0> <token1>\n";
        return 2;
    }

    try
    {
        const std::string model_dir = argv[1];
        const std::string output_path = argv[2];
        const std::vector<int32_t> input_ids_data = {
            parse_int32(argv[3], "token0"),
            parse_int32(argv[4], "token1"),
        };
        const std::vector<int32_t> positions_data = {0, 1};

        const std::filesystem::path model_path(model_dir);
        const std::filesystem::path weight_path = model_path / "model.safetensors";
        if (!std::filesystem::exists(model_path))
        {
            throw std::runtime_error("model directory does not exist: " + model_dir);
        }
        if (!std::filesystem::exists(weight_path))
        {
            throw std::runtime_error("model.safetensors does not exist: " + weight_path.string());
        }

        const tiny_llm::LlamaConfig config = tiny_llm::HFLlamaConfigLoader::load_from_dir(model_dir);
        for (int32_t token_id : input_ids_data)
        {
            if (token_id < 0 || token_id >= config.vocab_size)
            {
                throw std::runtime_error("token id is out of vocab range.");
            }
        }

        const tiny_llm::HFSafeTensorLoader loader =
            tiny_llm::HFSafeTensorLoader::from_file(weight_path.string());
        const tiny_llm::WeightMap weight_map = tiny_llm::WeightMap::from_safetensors(loader);

        tiny_llm::LlamaModel model(config, weight_map);
        model.allocate_buffers(static_cast<int32_t>(input_ids_data.size()));

        torch::Tensor input_ids = torch::from_blob(
            const_cast<int32_t*>(input_ids_data.data()),
            {static_cast<int64_t>(input_ids_data.size())},
            torch::TensorOptions().dtype(torch::kInt32)).clone();
        torch::Tensor positions = torch::from_blob(
            const_cast<int32_t*>(positions_data.data()),
            {static_cast<int64_t>(positions_data.size())},
            torch::TensorOptions().dtype(torch::kInt32)).clone();
        torch::Tensor logits = torch::zeros(
            {static_cast<int64_t>(input_ids_data.size()), config.vocab_size},
            torch::TensorOptions().dtype(torch::kFloat32));

        tiny_llm::ExecutionContext ctx(nullptr, nullptr, nullptr);
        model.forward_step(input_ids, positions, logits, ctx);
        logits = logits.contiguous();

        std::ofstream out(output_path, std::ios::binary);
        if (!out)
        {
            throw std::runtime_error("failed to open logits output file: " + output_path);
        }

        const int32_t batch_size = static_cast<int32_t>(logits.size(0));
        const int32_t vocab_size = static_cast<int32_t>(logits.size(1));
        write_exact(out, &batch_size, sizeof(batch_size));
        write_exact(out, &vocab_size, sizeof(vocab_size));
        write_exact(
            out,
            logits.data_ptr<float>(),
            static_cast<std::streamsize>(logits.numel() * static_cast<int64_t>(sizeof(float))));
    }
    catch (const std::exception& ex)
    {
        std::cerr << "llama_logits_dump failed: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
