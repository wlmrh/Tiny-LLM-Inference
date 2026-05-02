#include "tiny_llm/runtime/parallel_config.h"

#include <stdexcept>

namespace {

void expect_true(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void expect_throws(void (*fn)(), const char* message)
{
    try
    {
        fn();
    }
    catch (const std::runtime_error&)
    {
        return;
    }
    throw std::runtime_error(message);
}

void invalid_tensor_parallel()
{
    tiny_llm::ParallelConfig(
        tiny_llm::DeviceType::kCPU,
        0,
        2,
        1).validate();
}

void invalid_pipeline_parallel()
{
    tiny_llm::ParallelConfig(
        tiny_llm::DeviceType::kCPU,
        0,
        1,
        2).validate();
}

void invalid_cpu_device_id()
{
    tiny_llm::ParallelConfig(
        tiny_llm::DeviceType::kCPU,
        1,
        1,
        1).validate();
}

void invalid_cuda_device_id()
{
    tiny_llm::ParallelConfig::cuda(-1).validate();
}

} // namespace

int main()
{
    const tiny_llm::ParallelConfig cpu = tiny_llm::ParallelConfig::cpu();
    cpu.validate();
    expect_true(cpu.is_cpu(), "cpu config must report CPU.");
    expect_true(!cpu.is_cuda(), "cpu config must not report CUDA.");
    expect_true(cpu.device_id() == 0, "cpu device id must be 0.");
    expect_true(cpu.tensor_parallel_size() == 1, "cpu tensor parallel size must be 1.");
    expect_true(cpu.pipeline_parallel_size() == 1, "cpu pipeline parallel size must be 1.");
    expect_true(cpu.torch_device().is_cpu(), "cpu torch_device must be CPU.");

    const tiny_llm::ParallelConfig cuda = tiny_llm::ParallelConfig::cuda(0);
    cuda.validate();
    expect_true(cuda.is_cuda(), "cuda config must report CUDA.");
    expect_true(!cuda.is_cpu(), "cuda config must not report CPU.");
    expect_true(cuda.device_id() == 0, "cuda device id must be 0.");
    expect_true(cuda.torch_device().is_cuda(), "cuda torch_device must be CUDA.");
    expect_true(cuda.torch_device().index() == 0, "cuda torch_device index must be 0.");

    expect_throws(invalid_tensor_parallel, "tensor_parallel_size != 1 must throw.");
    expect_throws(invalid_pipeline_parallel, "pipeline_parallel_size != 1 must throw.");
    expect_throws(invalid_cpu_device_id, "CPU device_id != 0 must throw.");
    expect_throws(invalid_cuda_device_id, "CUDA device_id < 0 must throw.");

    return 0;
}
