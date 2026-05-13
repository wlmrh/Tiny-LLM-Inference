#include "tiny_llm/runtime/llm.h"

#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {
std::string model_dir_from_env()
{
    const char* model_dir = std::getenv("TINYLLM_HF_TINY_LLAMA_DIR");
    if (model_dir == nullptr || std::string(model_dir).empty())
    {
        return {};
    }
    return model_dir;
}
}

TEST(LLMOfflineIntegrationTest, GenerateReturnsFinishedOutputsForBatch)
{
    const std::string model_dir = model_dir_from_env();
    if (model_dir.empty())
    {
        GTEST_SKIP() << "TINYLLM_HF_TINY_LLAMA_DIR is not set.";
    }
    if (!std::filesystem::is_directory(model_dir))
    {
        GTEST_SKIP() << "Model directory does not exist: " << model_dir;
    }

    tiny_llm::LLMOptions options(model_dir);
    tiny_llm::UserSamplingParams sampling_params;
    sampling_params.temperature = 0.0f;
    sampling_params.max_tokens = 4;

    tiny_llm::LLM llm(options);
    const std::vector<tiny_llm::CompletionOutput> outputs =
        llm.generate(std::vector<std::string>{"hello", "tiny llm inference"}, sampling_params);
    ASSERT_EQ(outputs.size(), 2u);
    for (const tiny_llm::CompletionOutput& output : outputs)
    {
        EXPECT_TRUE(output.finished);
        EXPECT_FALSE(output.token_ids.empty());
    }
}
