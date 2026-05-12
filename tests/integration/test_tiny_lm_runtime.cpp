#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "tiny_llm/runtime/llm.h"

#ifndef TINYLLM_SOURCE_DIR
#error "TINYLLM_SOURCE_DIR must be defined by CMake for test assets."
#endif

namespace {

std::filesystem::path expand_user_path(const std::string& path)
{
    if (path.empty() || path[0] != '~')
    {
        return std::filesystem::path(path);
    }

    const char* home = std::getenv("HOME");
    if (home == nullptr || *home == '\0')
    {
        return std::filesystem::path(path);
    }

    if (path.size() == 1)
    {
        return std::filesystem::path(home);
    }
    if (path[1] == '/')
    {
        return std::filesystem::path(home) / path.substr(2);
    }

    return std::filesystem::path(path);
}

std::string resolve_model_dir(int argc, char** argv)
{
    if (argc > 1)
    {
        return expand_user_path(argv[1]).string();
    }
    if (const char* env_hf_dir = std::getenv("TINYLLM_HF_TINY_LLAMA_DIR"))
    {
        return expand_user_path(env_hf_dir).string();
    }

    std::cerr << "TINYLLM_HF_TINY_LLAMA_DIR is not set; skipping tiny LLM runtime integration test.\n";
    return {};
}

bool has_required_model_files(const std::filesystem::path& model_dir)
{
    if (!std::filesystem::is_directory(model_dir))
    {
        std::cerr << "model directory does not exist: " << model_dir << "\n";
        return false;
    }
    if (!std::filesystem::exists(model_dir / "config.json"))
    {
        std::cerr << "missing config.json under: " << model_dir << "\n";
        return false;
    }
    if (!std::filesystem::exists(model_dir / "model.safetensors"))
    {
        std::cerr << "missing model.safetensors under: " << model_dir << "\n";
        return false;
    }
    if (!std::filesystem::exists(model_dir / "tokenizer.json")
        && !std::filesystem::exists(model_dir / "tokenizer.model"))
    {
        std::cerr << "missing tokenizer.json or tokenizer.model under: " << model_dir << "\n";
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    const std::string hf_model_dir = resolve_model_dir(argc, argv);
    if (hf_model_dir.empty())
    {
        return 77;
    }
    if (!has_required_model_files(hf_model_dir))
    {
        return 77;
    }

    try
    {
#if TINYLLM_ENABLE_CUDA
        tiny_llm::LLMOptions options(hf_model_dir, tiny_llm::ParallelConfig::cuda(0));
#else
        tiny_llm::LLMOptions options(hf_model_dir);
#endif

        tiny_llm::LLMSamplingParams sampling_params;
        sampling_params.temperature = 0.0f;
        sampling_params.max_tokens = 8;

        tiny_llm::LLM llm(options);
        const std::vector<std::string> prompts = {"hello", "tiny llm inference"};
        std::vector<int32_t> stream_event_counts(prompts.size(), 0);
        std::vector<std::string> streamed_text(prompts.size());

        const std::vector<tiny_llm::CompletionOutput> outputs =
            llm.generate_stream(prompts, sampling_params, [&](const tiny_llm::CompletionStreamOutput& output) {
                assert(output.prompt_index < prompts.size());
                assert(output.prompt == prompts[output.prompt_index]);
                assert(output.token_id >= 0);
                ++stream_event_counts[output.prompt_index];
                streamed_text[output.prompt_index] += output.delta_text;
            });

        assert(outputs.size() == prompts.size());
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            const tiny_llm::CompletionOutput& output = outputs[i];
            assert(output.prompt == prompts[i]);
            assert(output.finished);
            assert(!output.text.empty());
            assert(!output.token_ids.empty());
            assert(stream_event_counts[i] > 0);
            assert(!streamed_text[i].empty());

            std::cout << "[test_tiny_lm_runtime] prompt: " << output.prompt << "\n";
            std::cout << "[test_tiny_lm_runtime] streamed: " << streamed_text[i] << "\n";
            std::cout << "[test_tiny_lm_runtime] output: " << output.text << "\n";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }

    return 0;
}
