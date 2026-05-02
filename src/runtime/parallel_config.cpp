#include "tiny_llm/runtime/parallel_config.h"

#include <stdexcept>

namespace tiny_llm {

ParallelConfig::ParallelConfig(DeviceType device_type,
                               int32_t device_id,
                               int32_t tensor_parallel_size,
                               int32_t pipeline_parallel_size)
    : device_type_(device_type),
      device_id_(device_id),
      tensor_parallel_size_(tensor_parallel_size),
      pipeline_parallel_size_(pipeline_parallel_size)
{
}

ParallelConfig ParallelConfig::cpu()
{
    return ParallelConfig(DeviceType::kCPU, 0, 1, 1);
}

ParallelConfig ParallelConfig::cuda(int32_t device_id)
{
    return ParallelConfig(DeviceType::kCUDA, device_id, 1, 1);
}

c10::Device ParallelConfig::torch_device() const
{
    validate();
    if (device_type_ == DeviceType::kCPU)
    {
        return c10::Device(c10::DeviceType::CPU);
    }
    if (device_type_ == DeviceType::kCUDA)
    {
        return c10::Device(c10::DeviceType::CUDA, device_id_);
    }
    throw std::runtime_error("ParallelConfig::torch_device: unsupported device type.");
}

bool ParallelConfig::operator==(const ParallelConfig& other) const
{
    return device_type_ == other.device_type_
        && device_id_ == other.device_id_
        && tensor_parallel_size_ == other.tensor_parallel_size_
        && pipeline_parallel_size_ == other.pipeline_parallel_size_;
}

void ParallelConfig::validate() const
{
    if (tensor_parallel_size_ != 1)
    {
        throw std::runtime_error("ParallelConfig: tensor_parallel_size must be 1 in v1.");
    }
    if (pipeline_parallel_size_ != 1)
    {
        throw std::runtime_error("ParallelConfig: pipeline_parallel_size must be 1 in v1.");
    }

    if (device_type_ == DeviceType::kCPU)
    {
        if (device_id_ != 0)
        {
            throw std::runtime_error("ParallelConfig: CPU device_id must be 0.");
        }
        return;
    }

    if (device_type_ == DeviceType::kCUDA)
    {
        if (device_id_ < 0)
        {
            throw std::runtime_error("ParallelConfig: CUDA device_id must be non-negative.");
        }
        return;
    }

    throw std::runtime_error("ParallelConfig: unsupported device type.");
}

} // namespace tiny_llm
