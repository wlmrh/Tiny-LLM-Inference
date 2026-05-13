#include "tiny_llm/core/context.h"
#include "tiny_llm/models/modules/linear.h"

#include <gtest/gtest.h>

TEST(LinearModuleTest, ComputesOutInProjection)
{
    tiny_llm::ExecutionContext ctx(nullptr, nullptr, nullptr);

    tiny_llm::Tensor input = torch::tensor(
        {{1.0f, 2.0f, 3.0f},
         {4.0f, 5.0f, 6.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor weight_out_in = torch::tensor(
        {{1.0f, 0.0f, 1.0f},
         {0.0f, 1.0f, 1.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));

    tiny_llm::modules::Linear linear(3, 2);
    linear.bind_weight(weight_out_in, tiny_llm::modules::WeightLayout::kOutIn);

    tiny_llm::Tensor output = torch::empty({2, 2}, torch::TensorOptions().dtype(torch::kFloat32));
    linear.forward(input, output, ctx);
    const float* output_ptr = output.data_ptr<float>();
    EXPECT_NEAR(output_ptr[0], 4.0f, 1e-5f);
    EXPECT_NEAR(output_ptr[1], 5.0f, 1e-5f);
    EXPECT_NEAR(output_ptr[2], 10.0f, 1e-5f);
    EXPECT_NEAR(output_ptr[3], 11.0f, 1e-5f);

    tiny_llm::Tensor returned_output = linear.forward(input, ctx);
    const float* returned_ptr = returned_output.data_ptr<float>();
    EXPECT_NEAR(returned_ptr[0], 4.0f, 1e-5f);
    EXPECT_NEAR(returned_ptr[3], 11.0f, 1e-5f);
}

TEST(LinearModuleTest, ComputesStackedWeightsWithBias)
{
    tiny_llm::ExecutionContext ctx(nullptr, nullptr, nullptr);
    tiny_llm::Tensor input = torch::tensor(
        {{1.0f, 2.0f, 3.0f},
         {4.0f, 5.0f, 6.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor w0 = torch::tensor({{1.0f, 1.0f, 0.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor w1 = torch::tensor(
        {{0.0f, 1.0f, 0.0f},
         {0.0f, 0.0f, 1.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor b1 = torch::tensor({10.0f, 20.0f}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::modules::StackedWeightDesc descs[2] = {
        {nullptr, 1, 3, 0, tiny_llm::modules::WeightLayout::kOutIn, w0},
        {nullptr, 2, 3, 1, tiny_llm::modules::WeightLayout::kOutIn, w1, b1},
    };

    tiny_llm::modules::Linear stacked_linear(3, 3);
    stacked_linear.bind_stacked_weights(descs, 2);
    tiny_llm::Tensor stacked_output = torch::empty({2, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    stacked_linear.forward(input, stacked_output, ctx);

    const float* ptr = stacked_output.data_ptr<float>();
    EXPECT_NEAR(ptr[0], 3.0f, 1e-5f);
    EXPECT_NEAR(ptr[1], 12.0f, 1e-5f);
    EXPECT_NEAR(ptr[2], 23.0f, 1e-5f);
    EXPECT_NEAR(ptr[3], 9.0f, 1e-5f);
    EXPECT_NEAR(ptr[4], 15.0f, 1e-5f);
    EXPECT_NEAR(ptr[5], 26.0f, 1e-5f);
}
