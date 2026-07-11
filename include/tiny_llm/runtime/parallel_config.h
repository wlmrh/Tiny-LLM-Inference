#pragma once

#include <cstdint>

namespace c10
{
class Device;
}

namespace tiny_llm
{

enum class DeviceType
{
    kCPU = 0,
    kCUDA = 1,
};

class ParallelConfig
{
  public:
    ParallelConfig() = default;

    static ParallelConfig cpu();
    static ParallelConfig cuda(int32_t device_id = 0);

    DeviceType device_type() const
    {
        return device_type_;
    }
    int32_t device_id() const
    {
        return device_id_;
    }
    bool is_cpu() const
    {
        return device_type_ == DeviceType::kCPU;
    }
    bool is_cuda() const
    {
        return device_type_ == DeviceType::kCUDA;
    }
    bool operator==(const ParallelConfig &other) const;
    bool operator!=(const ParallelConfig &other) const
    {
        return !(*this == other);
    }

    c10::Device torch_device() const;
    void validate() const;

  private:
    ParallelConfig(DeviceType device_type, int32_t device_id);

    DeviceType device_type_ = DeviceType::kCPU;
    int32_t device_id_ = 0;
};

} // namespace tiny_llm
