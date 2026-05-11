#include "tiny_llm/core/context.h"
#include "tiny_llm/models/modules/linear.h"

#include <cmath>
#include <stdexcept>

namespace {

void expect_near(float actual, float expected, const char* message)
{
    if (std::fabs(actual - expected) > 1e-5f)
    {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    tiny_llm::ExecutionContext ctx(nullptr, nullptr, nullptr);

    tiny_llm::Tensor input = torch::tensor(
        {{1.0f, 2.0f, 3.0f},
         {4.0f, 5.0f, 6.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));

    tiny_llm::Tensor output = torch::empty(
        {2, 2},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor weight_out_in = torch::tensor(
        {{1.0f, 0.0f, 1.0f},
         {0.0f, 1.0f, 1.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));

    tiny_llm::modules::Linear linear(3, 2);
    linear.bind_weight(weight_out_in, tiny_llm::modules::WeightLayout::kOutIn);
    linear.forward(input, output, ctx);

    const float* output_ptr = output.data_ptr<float>();
    expect_near(output_ptr[0], 4.0f, "single kOutIn row 0 col 0 mismatch.");
    expect_near(output_ptr[1], 5.0f, "single kOutIn row 0 col 1 mismatch.");
    expect_near(output_ptr[2], 10.0f, "single kOutIn row 1 col 0 mismatch.");
    expect_near(output_ptr[3], 11.0f, "single kOutIn row 1 col 1 mismatch.");

    tiny_llm::Tensor returned_output = linear.forward(input, ctx);
    const float* returned_ptr = returned_output.data_ptr<float>();
    expect_near(returned_ptr[0], 4.0f, "returned single kOutIn row 0 col 0 mismatch.");
    expect_near(returned_ptr[3], 11.0f, "returned single kOutIn row 1 col 1 mismatch.");

    tiny_llm::Tensor stacked_output = torch::empty(
        {2, 3},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor w0 = torch::tensor(
        {{1.0f, 1.0f, 0.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor w1 = torch::tensor(
        {{0.0f, 1.0f, 0.0f},
         {0.0f, 0.0f, 1.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::modules::StackedWeightDesc descs[2] = {
        {nullptr, 1, 3, 0, tiny_llm::modules::WeightLayout::kOutIn, w0},
        {nullptr, 2, 3, 1, tiny_llm::modules::WeightLayout::kOutIn, w1},
    };

    tiny_llm::modules::Linear stacked_linear(3, 3);
    stacked_linear.bind_stacked_weights(descs, 2);
    stacked_linear.forward(input, stacked_output, ctx);

    const float* stacked_ptr = stacked_output.data_ptr<float>();
    expect_near(stacked_ptr[0], 3.0f, "stacked row 0 col 0 mismatch.");
    expect_near(stacked_ptr[1], 2.0f, "stacked row 0 col 1 mismatch.");
    expect_near(stacked_ptr[2], 3.0f, "stacked row 0 col 2 mismatch.");
    expect_near(stacked_ptr[3], 9.0f, "stacked row 1 col 0 mismatch.");
    expect_near(stacked_ptr[4], 5.0f, "stacked row 1 col 1 mismatch.");
    expect_near(stacked_ptr[5], 6.0f, "stacked row 1 col 2 mismatch.");

    return 0;
}
