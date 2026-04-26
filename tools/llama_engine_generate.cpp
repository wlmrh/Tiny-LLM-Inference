#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tiny_llm/core/allocator.h"
#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/runtime/engine.h"
#include "tiny_llm/runtime/tokenizer.h"

namespace {

std::filesystem::path expand_user_path(const std::string& path)
{
    if (path.empty() || path[0] != '~')
    {
        return path;
    }

    const char* home = std::getenv("HOME");
    if (home == nullptr)
    {
        return path;
    }
    if (path.size() == 1)
    {
        return home;
    }
    if (path[1] == '/')
    {
        return std::filesystem::path(home) / path.substr(2);
    }
    return path;
}

int32_t parse_int32(const char* text, const char* name)
{
    try
    {
        size_t consumed = 0;
        const long value = std::stol(text, &consumed);
        if (consumed != std::string(text).size() || value < INT32_MIN || value > INT32_MAX)
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

size_t llama_kv_block_bytes(const tiny_llm::LlamaConfig& config, int32_t block_size_tokens)
{
    if (block_size_tokens <= 0 || config.head_dim <= 0 || config.num_key_value_heads <= 0)
    {
        throw std::runtime_error("invalid KV cache dimensions.");
    }
    const size_t kv_hidden_size =
        static_cast<size_t>(config.num_key_value_heads) * static_cast<size_t>(config.head_dim);
    return 2 * static_cast<size_t>(block_size_tokens) * kv_hidden_size * sizeof(float);
}

std::string json_escape(const std::string& text)
{
    std::ostringstream out;
    for (unsigned char ch : text)
    {
        switch (ch)
        {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (ch < 0x20)
            {
                out << "\\u";
                const char* hex = "0123456789abcdef";
                out << "00" << hex[(ch >> 4) & 0x0f] << hex[ch & 0x0f];
            }
            else
            {
                out << static_cast<char>(ch);
            }
            break;
        }
    }
    return out.str();
}

void print_json_result(const std::string& prompt, const tiny_llm::UserOutput& output)
{
    std::cout << "{\"prompt\":\"" << json_escape(prompt) << "\",";
    std::cout << "\"output\":\"" << json_escape(output.text) << "\",";
    std::cout << "\"finish_reason\":\"" << json_escape(output.finish_reason) << "\",";
    std::cout << "\"generated_token_ids\":[";
    for (size_t i = 0; i < output.generated_token_ids.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ",";
        }
        std::cout << output.generated_token_ids[i];
    }
    std::cout << "]}\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "usage: " << argv[0] << " <model_dir> <max_new_tokens> <prompt> [prompt...]\n";
        return 2;
    }

    try
    {
        const std::filesystem::path model_dir = expand_user_path(argv[1]);
        const int32_t max_new_tokens = parse_int32(argv[2], "max_new_tokens");
        if (max_new_tokens <= 0)
        {
            throw std::runtime_error("max_new_tokens must be positive.");
        }
        if (!std::filesystem::exists(model_dir / "config.json")
            || !std::filesystem::exists(model_dir / "model.safetensors")
            || !std::filesystem::exists(model_dir / "tokenizer.json"))
        {
            throw std::runtime_error("model_dir must contain config.json, model.safetensors, and tokenizer.json.");
        }

        const tiny_llm::LlamaConfig hf_config =
            tiny_llm::HFLlamaConfigLoader::load_from_dir(model_dir.string());
        tiny_llm::HFLlamaTokenizer tokenizer =
            tiny_llm::HFLlamaTokenizer::from_model_dir(model_dir.string());
        assert(tokenizer.vocab_size() == hf_config.vocab_size);

        tiny_llm::StackAllocator allocator(16 * 1024 * 1024);
        constexpr int32_t kBlockSizeTokens = 16;
        constexpr size_t kNumBlocks = 256;
        const size_t kBlockBytes = llama_kv_block_bytes(hf_config, kBlockSizeTokens);
        void* kv_pool = std::malloc(kNumBlocks * kBlockBytes);
        if (kv_pool == nullptr)
        {
            throw std::runtime_error("failed to allocate KV metadata pool.");
        }

        tiny_llm::EngineArgs engine_args;
        engine_args.tokenizer = &tokenizer;
        engine_args.model_type = tiny_llm::EngineModelType::kHFLlamaSafeTensor;
        engine_args.hf_model_dir = model_dir.string();
        engine_args.hf_weight_file = "model.safetensors";
        engine_args.execution_stream = nullptr;
        engine_args.workspace = &allocator;
        engine_args.max_batch_size = 16;
        engine_args.kv_num_layers = hf_config.num_hidden_layers;
        engine_args.kv_block_size_tokens = kBlockSizeTokens;
        engine_args.kv_num_blocks = kNumBlocks;
        engine_args.kv_block_size_bytes = kBlockBytes;
        engine_args.kv_memory_pool = kv_pool;
        engine_args.max_generated_tokens = max_new_tokens;

        tiny_llm::LLMEngine engine(engine_args);
        tiny_llm::UserSamplingParams sampling_params;
        sampling_params.temperature = 0.0f;
        sampling_params.top_p = 1.0f;
        sampling_params.top_k = 0;
        sampling_params.max_tokens = max_new_tokens;

        struct PendingPrompt {
            uint64_t request_id = 0;
            std::string prompt;
            tiny_llm::UserOutput last_output;
        };
        std::vector<PendingPrompt> prompts;
        prompts.reserve(static_cast<size_t>(argc - 3));
        for (int arg = 3; arg < argc; ++arg)
        {
            PendingPrompt pending;
            pending.prompt = argv[arg];
            pending.request_id = engine.add_request(pending.prompt, sampling_params);
            prompts.push_back(std::move(pending));
        }

        while (engine.has_unfinished_requests())
        {
            const std::vector<tiny_llm::UserOutput> outputs = engine.step();
            for (const tiny_llm::UserOutput& output : outputs)
            {
                for (PendingPrompt& pending : prompts)
                {
                    if (pending.request_id == output.internal_id)
                    {
                        pending.last_output = output;
                        break;
                    }
                }
            }
        }

        for (const PendingPrompt& pending : prompts)
        {
            print_json_result(pending.prompt, pending.last_output);
        }

        std::free(kv_pool);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "llama_engine_generate failed: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
