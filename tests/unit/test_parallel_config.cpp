#include "tiny_llm/runtime/parallel_config.h"

#include <gtest/gtest.h>
#include <stdexcept>

TEST(ParallelConfigTest, CpuConfigIsValid)
{
    const tiny_llm::ParallelConfig cpu = tiny_llm::ParallelConfig::cpu();
    EXPECT_NO_THROW(cpu.validate());
    EXPECT_TRUE(cpu.is_cpu());
    EXPECT_FALSE(cpu.is_cuda());
    EXPECT_EQ(cpu.device_id(), 0);
    EXPECT_EQ(cpu.tensor_parallel_size(), 1);
    EXPECT_EQ(cpu.pipeline_parallel_size(), 1);
    EXPECT_TRUE(cpu.torch_device().is_cpu());
}

TEST(ParallelConfigTest, CudaConfigCarriesDeviceIndex)
{
    const tiny_llm::ParallelConfig cuda = tiny_llm::ParallelConfig::cuda(0);
    EXPECT_NO_THROW(cuda.validate());
    EXPECT_TRUE(cuda.is_cuda());
    EXPECT_FALSE(cuda.is_cpu());
    EXPECT_EQ(cuda.device_id(), 0);
    EXPECT_TRUE(cuda.torch_device().is_cuda());
    EXPECT_EQ(cuda.torch_device().index(), 0);
}

TEST(ParallelConfigTest, RejectsUnsupportedOrInvalidLayouts)
{
    EXPECT_THROW(tiny_llm::ParallelConfig(tiny_llm::DeviceType::kCPU, 0, 2, 1).validate(), std::runtime_error);
    EXPECT_THROW(tiny_llm::ParallelConfig(tiny_llm::DeviceType::kCPU, 0, 1, 2).validate(), std::runtime_error);
    EXPECT_THROW(tiny_llm::ParallelConfig(tiny_llm::DeviceType::kCPU, 1, 1, 1).validate(), std::runtime_error);
    EXPECT_THROW(tiny_llm::ParallelConfig::cuda(-1).validate(), std::runtime_error);
}
