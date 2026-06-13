#include "tiny_llm/runtime/sampler.h"

#include <gtest/gtest.h>

TEST(SamplerTest, GreedyPenaltyOneMatchesArgmax)
{
    tiny_llm::Tensor logits = torch::tensor(
        {{0.0f, 2.0f, 3.0f}, {5.0f, 4.0f, 1.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<int32_t> rows = {0, 1};
    std::vector<std::vector<int32_t>> histories = {{2}, {0}};
    std::vector<tiny_llm::SamplingParams> params(2);
    params[0].repetition_penalty = 1.0f;
    params[1].repetition_penalty = 1.0f;

    const std::vector<int32_t> sampled = tiny_llm::sample_greedy_rows(logits, rows, 3, &histories, &params);
    EXPECT_EQ(sampled, std::vector<int32_t>({2, 0}));
}

TEST(SamplerTest, AppliesPositiveRepetitionPenaltyBeforeArgmax)
{
    tiny_llm::Tensor logits = torch::tensor(
        {{0.0f, 5.0f, 4.6f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<int32_t> rows = {0};
    std::vector<std::vector<int32_t>> histories = {{1}};
    std::vector<tiny_llm::SamplingParams> params(1);
    params[0].repetition_penalty = 2.0f;

    const std::vector<int32_t> sampled = tiny_llm::sample_greedy_rows(logits, rows, 3, &histories, &params);
    EXPECT_EQ(sampled[0], 2);
}

TEST(SamplerTest, AppliesNegativeRepetitionPenaltyBeforeArgmax)
{
    tiny_llm::Tensor logits = torch::tensor(
        {{-2.0f, -0.5f, -0.8f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<int32_t> rows = {0};
    std::vector<std::vector<int32_t>> histories = {{1}};
    std::vector<tiny_llm::SamplingParams> params(1);
    params[0].repetition_penalty = 2.0f;

    const std::vector<int32_t> sampled = tiny_llm::sample_greedy_rows(logits, rows, 3, &histories, &params);
    EXPECT_EQ(sampled[0], 2);
}

TEST(SamplerTest, PenalizesPromptHistoryBeforeAnyGeneratedToken)
{
    tiny_llm::Tensor logits = torch::tensor({{0.0f, 9.0f, 8.5f}}, torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<int32_t> rows = {0};
    std::vector<std::vector<int32_t>> histories = {{1, 1}};
    std::vector<tiny_llm::SamplingParams> params(1);
    params[0].repetition_penalty = 1.1f;

    const std::vector<int32_t> sampled = tiny_llm::sample_greedy_rows(logits, rows, 3, &histories, &params);
    EXPECT_EQ(sampled[0], 2);
}

TEST(SamplerTest, AppliesRepetitionRewardBelowOneBeforeArgmax)
{
    tiny_llm::Tensor logits = torch::tensor(
        {{0.0f, 4.0f, 4.3f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<int32_t> rows = {0};
    std::vector<std::vector<int32_t>> histories = {{1}};
    std::vector<tiny_llm::SamplingParams> params(1);
    params[0].repetition_penalty = 0.5f;

    const std::vector<int32_t> sampled = tiny_llm::sample_greedy_rows(logits, rows, 3, &histories, &params);
    EXPECT_EQ(sampled[0], 1);
}

TEST(SamplerTest, RejectsMismatchedPenaltyMetadata)
{
    tiny_llm::Tensor logits = torch::tensor({{0.0f, 1.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<int32_t> rows = {0};
    std::vector<std::vector<int32_t>> histories;
    std::vector<tiny_llm::SamplingParams> params(1);
    params[0].repetition_penalty = 1.1f;

    EXPECT_THROW(tiny_llm::sample_greedy_rows(logits, rows, 2, &histories, &params), std::runtime_error);
}

TEST(SamplerTest, RejectsHistoryTokenOutsideVocabulary)
{
    tiny_llm::Tensor logits = torch::tensor({{0.0f, 1.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<int32_t> rows = {0};
    std::vector<std::vector<int32_t>> histories = {{2}};
    std::vector<tiny_llm::SamplingParams> params(1);
    params[0].repetition_penalty = 1.1f;

    EXPECT_THROW(tiny_llm::sample_greedy_rows(logits, rows, 2, &histories, &params), std::runtime_error);
}

TEST(SamplerTest, TopKOneSamplingAlwaysSelectsTopToken)
{
    tiny_llm::Tensor logits = torch::tensor({{0.0f, 1.0f, 5.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<int32_t> rows = {0};
    std::vector<tiny_llm::SamplingParams> params(1);
    params[0].temperature = 1.0f;
    params[0].top_k = 1;
    params[0].seed = 123;

    const std::vector<int32_t> sampled = tiny_llm::sample_greedy_rows(logits, rows, 3, nullptr, &params);
    EXPECT_EQ(sampled[0], 2);
}

TEST(SamplerTest, TopPFilteringKeepsAtLeastMostLikelyToken)
{
    tiny_llm::Tensor logits = torch::tensor({{10.0f, 9.0f, 0.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<int32_t> rows = {0};
    std::vector<tiny_llm::SamplingParams> params(1);
    params[0].temperature = 1.0f;
    params[0].top_p = 0.5f;
    params[0].seed = 123;

    const std::vector<int32_t> sampled = tiny_llm::sample_greedy_rows(logits, rows, 3, nullptr, &params);
    EXPECT_EQ(sampled[0], 0);
}

TEST(SamplerTest, SeededSamplingIsReproducible)
{
    tiny_llm::Tensor logits = torch::tensor({{0.0f, 0.2f, 0.4f, 0.6f}}, torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<int32_t> rows = {0};
    std::vector<std::vector<int32_t>> histories = {{1, 2}};
    std::vector<tiny_llm::SamplingParams> params(1);
    params[0].temperature = 0.8f;
    params[0].seed = 98765;
    std::vector<uint64_t> request_ids = {42};

    const std::vector<int32_t> first =
        tiny_llm::sample_greedy_rows(logits, rows, 4, &histories, &params, &request_ids);
    const std::vector<int32_t> second =
        tiny_llm::sample_greedy_rows(logits, rows, 4, &histories, &params, &request_ids);
    EXPECT_EQ(first, second);
    EXPECT_GE(first[0], 0);
    EXPECT_LT(first[0], 4);
}


#if TINYLLM_ENABLE_CUDA
TEST(SamplerTest, CudaRepetitionPenaltyMatchesCpuForNonDenseRows)
{
    if (!torch::cuda::is_available())
    {
        GTEST_SKIP() << "CUDA is not available.";
    }

    tiny_llm::Tensor logits_cpu = torch::tensor(
        {{0.0f, 5.0f, 4.6f, -1.0f},
         {9.0f, 1.0f, 0.0f, 2.0f},
         {3.0f, 2.0f, 4.0f, 3.8f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor logits_cuda = logits_cpu.to(torch::kCUDA);
    std::vector<int32_t> rows = {0, 2};
    std::vector<std::vector<int32_t>> histories = {{1, 1}, {2}};
    std::vector<tiny_llm::SamplingParams> params(2);
    params[0].repetition_penalty = 2.0f;
    params[1].repetition_penalty = 0.5f;

    const std::vector<int32_t> cpu_sampled =
        tiny_llm::sample_greedy_rows(logits_cpu, rows, 4, &histories, &params);
    const std::vector<int32_t> cuda_sampled =
        tiny_llm::sample_greedy_rows(logits_cuda, rows, 4, &histories, &params);

    EXPECT_EQ(cuda_sampled, cpu_sampled);
}

TEST(SamplerTest, CudaNonGreedySamplingMatchesCpuFallback)
{
    if (!torch::cuda::is_available())
    {
        GTEST_SKIP() << "CUDA is not available.";
    }

    tiny_llm::Tensor logits_cpu = torch::tensor(
        {{0.0f, 1.0f, 5.0f},
         {2.0f, 0.0f, 1.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor logits_cuda = logits_cpu.to(torch::kCUDA);
    std::vector<int32_t> rows = {0, 1};
    std::vector<tiny_llm::SamplingParams> params(2);
    params[0].temperature = 1.0f;
    params[0].top_k = 1;
    params[1].temperature = 1.0f;
    params[1].top_k = 1;

    const std::vector<int32_t> cpu_sampled =
        tiny_llm::sample_greedy_rows(logits_cpu, rows, 3, nullptr, &params);
    const std::vector<int32_t> cuda_sampled =
        tiny_llm::sample_greedy_rows(logits_cuda, rows, 3, nullptr, &params);

    EXPECT_EQ(cuda_sampled, cpu_sampled);
}
#endif
