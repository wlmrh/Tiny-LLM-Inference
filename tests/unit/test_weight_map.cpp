#include "tiny_llm/models/llama_weight_map.h"

#include <stdexcept>

namespace {

void expect_true(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    tiny_llm::WeightMap weight_map;
    torch::Tensor tensor = torch::arange(
        6,
        torch::TensorOptions().dtype(torch::kFloat32).device(c10::kCPU)).reshape({2, 3});

    weight_map.add_tensor("weight", tensor);
    expect_true(weight_map.contains("weight"), "WeightMap must contain inserted tensor.");
    expect_true(weight_map.get_tensor("weight") != nullptr, "WeightMap raw pointer must be non-null.");

    const tiny_llm::Tensor& view = weight_map.get_tensor_view("weight");
    expect_true(view.device().is_cpu(), "WeightMap tensor view must keep CPU device.");
    expect_true(tiny_llm::tensor_shape(view) == std::vector<int64_t>({2, 3}), "WeightMap tensor shape mismatch.");
    expect_true(tiny_llm::tensor_dtype(view) == tiny_llm::DType::kFloat32, "WeightMap tensor dtype mismatch.");

    const float* ptr = weight_map.get_tensor_as<float>("weight");
    expect_true(ptr[0] == 0.0f && ptr[5] == 5.0f, "WeightMap data pointer values mismatch.");

    return 0;
}
