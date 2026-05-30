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

    EXPECT_FLOAT_EQ(tiny_llm::apply_repetition_penalty_to_logit(5.0f, 2.0f), 2.5f);
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

    EXPECT_FLOAT_EQ(tiny_llm::apply_repetition_penalty_to_logit(-0.5f, 2.0f), -1.0f);
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

    EXPECT_FLOAT_EQ(tiny_llm::apply_repetition_penalty_to_logit(4.0f, 0.5f), 8.0f);
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
