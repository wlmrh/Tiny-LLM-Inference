#pragma once

#include <cstdint>

#include <c10/core/Device.h>

namespace tiny_llm {

enum class DeviceType {
    kCPU = 0,
    kCUDA = 1,
};

class ParallelConfig {
public:
    ParallelConfig() = default;
    ParallelConfig(DeviceType device_type,
                   int32_t device_id,
                   int32_t tensor_parallel_size,
                   int32_t pipeline_parallel_size);

    static ParallelConfig cpu();
    static ParallelConfig cuda(int32_t device_id = 0);

    DeviceType device_type() const { return device_type_; }
    int32_t device_id() const { return device_id_; }
    int32_t tensor_parallel_size() const { return tensor_parallel_size_; }
    int32_t pipeline_parallel_size() const { return pipeline_parallel_size_; }
    bool is_cpu() const { return device_type_ == DeviceType::kCPU; }
    bool is_cuda() const { return device_type_ == DeviceType::kCUDA; }
    bool operator==(const ParallelConfig& other) const;
    bool operator!=(const ParallelConfig& other) const { return !(*this == other); }

    c10::Device torch_device() const;
    void validate() const;

private:
    DeviceType device_type_ = DeviceType::kCPU;
    int32_t device_id_ = 0;
    int32_t tensor_parallel_size_ = 1;
    int32_t pipeline_parallel_size_ = 1;
};

} // namespace tiny_llm
