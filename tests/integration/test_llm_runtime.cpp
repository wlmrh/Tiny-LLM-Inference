#include "tiny_llm/runtime/llm.h"

#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {
std::string g_model_dir_arg;
}

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

std::string resolve_model_dir()
{
    if (!g_model_dir_arg.empty())
    {
        return expand_user_path(g_model_dir_arg).string();
    }
    if (const char* env_hf_dir = std::getenv("TINYLLM_HF_TINY_LLAMA_DIR"))
    {
        return expand_user_path(env_hf_dir).string();
    }
    return {};
}

bool has_required_model_files(const std::filesystem::path& model_dir, std::string& reason)
{
    if (model_dir.empty())
    {
        reason = "TINYLLM_HF_TINY_LLAMA_DIR is not set.";
        return false;
    }
    if (!std::filesystem::is_directory(model_dir))
    {
        reason = "Model directory does not exist: " + model_dir.string();
        return false;
    }
    if (!std::filesystem::exists(model_dir / "config.json"))
    {
        reason = "Missing config.json under: " + model_dir.string();
        return false;
    }
    if (!std::filesystem::exists(model_dir / "model.safetensors"))
    {
        reason = "Missing model.safetensors under: " + model_dir.string();
        return false;
    }
    if (!std::filesystem::exists(model_dir / "tokenizer.json")
        && !std::filesystem::exists(model_dir / "tokenizer.model"))
    {
        reason = "Missing tokenizer.json or tokenizer.model under: " + model_dir.string();
        return false;
    }
    return true;
}
}

TEST(LLMRuntimeIntegrationTest, GenerateCallbackReturnsEventsAndFinalOutputs)
{
    const std::filesystem::path model_dir(resolve_model_dir());
    std::string skip_reason;
    if (!has_required_model_files(model_dir, skip_reason))
    {
        GTEST_SKIP() << skip_reason;
    }

#if TINYLLM_ENABLE_CUDA
    tiny_llm::LLMOptions options(model_dir.string(), tiny_llm::ParallelConfig::cuda(0));
#else
    tiny_llm::LLMOptions options(model_dir.string());
#endif

    tiny_llm::LLMSamplingParams sampling_params;
    sampling_params.temperature = 0.0f;
    sampling_params.max_tokens = 8;

    tiny_llm::LLM llm(options);
    const std::vector<std::string> prompts = {"hello", "tiny llm inference"};
    std::vector<int32_t> stream_event_counts(prompts.size(), 0);
    std::vector<std::string> streamed_text(prompts.size());

    const std::vector<tiny_llm::CompletionOutput> outputs =
        llm.generate(prompts, sampling_params, [&](const tiny_llm::CompletionStreamOutput& output) {
            ASSERT_LT(output.prompt_index, prompts.size());
            EXPECT_EQ(output.prompt, prompts[output.prompt_index]);
            EXPECT_GE(output.token_id, 0);
            ++stream_event_counts[output.prompt_index];
            streamed_text[output.prompt_index] += output.delta_text;
        });

    ASSERT_EQ(outputs.size(), prompts.size());
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        EXPECT_EQ(outputs[i].prompt, prompts[i]);
        EXPECT_TRUE(outputs[i].finished);
        EXPECT_FALSE(outputs[i].text.empty());
        EXPECT_FALSE(outputs[i].token_ids.empty());
        EXPECT_GT(stream_event_counts[i], 0);
        EXPECT_FALSE(streamed_text[i].empty());
    }
}

int main(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg.rfind("--gtest", 0) != 0)
        {
            g_model_dir_arg = arg;
            break;
        }
    }

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
