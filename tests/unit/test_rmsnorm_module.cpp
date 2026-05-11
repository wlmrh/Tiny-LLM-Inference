#include "tiny_llm/core/context.h"
#include "tiny_llm/models/modules/rmsnorm.h"

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

float rmsnorm_expected(float value, float rms, float weight)
{
    return value * rms * weight;
}

} // namespace

int main()
{
    tiny_llm::ExecutionContext ctx(nullptr, nullptr, nullptr);

    tiny_llm::Tensor input = torch::tensor(
        {{3.0f, 4.0f},
         {0.0f, 2.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor weight = torch::tensor(
        {1.0f, 0.5f},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor output = torch::empty_like(input);

    tiny_llm::modules::RMSNorm rmsnorm(2, 1e-6f);
    rmsnorm.bind_weights(weight);
    rmsnorm.forward(input, output, ctx);

    const float row0_rms = 1.0f / std::sqrt((9.0f + 16.0f) / 2.0f + 1e-6f);
    const float row1_rms = 1.0f / std::sqrt((0.0f + 4.0f) / 2.0f + 1e-6f);
    const float* output_ptr = output.data_ptr<float>();
    expect_near(output_ptr[0], rmsnorm_expected(3.0f, row0_rms, 1.0f), "row 0 col 0 mismatch.");
    expect_near(output_ptr[1], rmsnorm_expected(4.0f, row0_rms, 0.5f), "row 0 col 1 mismatch.");
    expect_near(output_ptr[2], rmsnorm_expected(0.0f, row1_rms, 1.0f), "row 1 col 0 mismatch.");
    expect_near(output_ptr[3], rmsnorm_expected(2.0f, row1_rms, 0.5f), "row 1 col 1 mismatch.");

    tiny_llm::Tensor returned_output = rmsnorm.forward(input, ctx);
    const float* returned_ptr = returned_output.data_ptr<float>();
    expect_near(returned_ptr[0], output_ptr[0], "returned row 0 col 0 mismatch.");
    expect_near(returned_ptr[3], output_ptr[3], "returned row 1 col 1 mismatch.");

    return 0;
}
