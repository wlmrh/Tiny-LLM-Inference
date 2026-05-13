#include "tiny_llm/models/llama_weight_map.h"

#include <gtest/gtest.h>

TEST(WeightMapTest, StoresAndReturnsTensorViews)
{
    tiny_llm::WeightMap weight_map;
    torch::Tensor tensor = torch::arange(
        6,
        torch::TensorOptions().dtype(torch::kFloat32).device(c10::kCPU)).reshape({2, 3});

    weight_map.add_tensor("weight", tensor);
    EXPECT_TRUE(weight_map.contains("weight"));
    EXPECT_NE(weight_map.get_tensor("weight"), nullptr);

    const tiny_llm::Tensor& view = weight_map.get_tensor_view("weight");
    EXPECT_TRUE(view.device().is_cpu());
    EXPECT_EQ(tiny_llm::tensor_shape(view), std::vector<int64_t>({2, 3}));
    EXPECT_EQ(tiny_llm::tensor_dtype(view), tiny_llm::DType::kFloat32);

    const float* ptr = weight_map.get_tensor_as<float>("weight");
    EXPECT_FLOAT_EQ(ptr[0], 0.0f);
    EXPECT_FLOAT_EQ(ptr[5], 5.0f);
}
