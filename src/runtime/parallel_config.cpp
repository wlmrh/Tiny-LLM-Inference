#include "tiny_llm/runtime/parallel_config.h"

#include <c10/core/Device.h>

#include <stdexcept>

namespace tiny_llm {

ParallelConfig::ParallelConfig(DeviceType device_type, int32_t device_id)
    : device_type_(device_type),
      device_id_(device_id)
{
}

ParallelConfig ParallelConfig::cpu()
{
    return ParallelConfig(DeviceType::kCPU, 0);
}

ParallelConfig ParallelConfig::cuda(int32_t device_id)
{
    return ParallelConfig(DeviceType::kCUDA, device_id);
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
        && device_id_ == other.device_id_;
}

void ParallelConfig::validate() const
{
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
