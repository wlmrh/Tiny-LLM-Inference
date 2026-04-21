#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

#include "utils/cuda_compat.h"

namespace tiny_llm {

/**
 * @enum DType
 * @brief Supported element data types for tensor operations.
 */
enum class DType { kFloat16, kFloat32, kInt32 };

using Tensor = torch::Tensor;

inline c10::ScalarType to_torch_scalar_type(DType dtype)
{
    switch (dtype)
    {
        case DType::kFloat16:
            return c10::ScalarType::Half;
        case DType::kFloat32:
            return c10::ScalarType::Float;
        case DType::kInt32:
            return c10::ScalarType::Int;
    }

    throw std::runtime_error("to_torch_scalar_type: unsupported DType.");
}

inline DType from_torch_scalar_type(c10::ScalarType scalar_type)
{
    switch (scalar_type)
    {
        case c10::ScalarType::Half:
            return DType::kFloat16;
        case c10::ScalarType::Float:
            return DType::kFloat32;
        case c10::ScalarType::Int:
            return DType::kInt32;
        default:
            throw std::runtime_error("from_torch_scalar_type: unsupported torch scalar type.");
    }
}

inline DType tensor_dtype(const Tensor& tensor)
{
    if (!tensor.defined())
    {
        throw std::runtime_error("tensor_dtype: tensor must be defined.");
    }
    return from_torch_scalar_type(tensor.scalar_type());
}

inline std::vector<int64_t> tensor_shape(const Tensor& tensor)
{
    if (!tensor.defined())
    {
        return {};
    }

    const auto sizes = tensor.sizes();
    return std::vector<int64_t>(sizes.begin(), sizes.end());
}

inline void* tensor_data(const Tensor& tensor)
{
    if (!tensor.defined())
    {
        return nullptr;
    }
    return tensor.data_ptr();
}

inline size_t tensor_numel(const Tensor& tensor)
{
    if (!tensor.defined())
    {
        return 0;
    }
    return static_cast<size_t>(tensor.numel());
}

inline c10::Device infer_blob_device(const void* data)
{
#if TINYLLM_ENABLE_CUDA
    if (data != nullptr)
    {
        cudaPointerAttributes attrs{};
        const cudaError_t status = cudaPointerGetAttributes(&attrs, data);
        if (status == cudaSuccess)
        {
#if CUDART_VERSION >= 10000
            if (attrs.type == cudaMemoryTypeDevice || attrs.type == cudaMemoryTypeManaged)
            {
                const int device = attrs.device >= 0 ? attrs.device : 0;
                return c10::Device(c10::DeviceType::CUDA, device);
            }
#else
            if (attrs.memoryType == cudaMemoryTypeDevice)
            {
                return c10::Device(c10::DeviceType::CUDA, 0);
            }
#endif
        }

        (void)cudaGetLastError();
    }
#endif

    return c10::Device(c10::DeviceType::CPU);
}

inline Tensor make_tensor_from_blob(void* data,
                                    const std::vector<int64_t>& shape,
                                    DType dtype)
{
    const auto options = torch::TensorOptions()
        .dtype(to_torch_scalar_type(dtype))
        .device(infer_blob_device(data));

    if (data == nullptr)
    {
        return torch::empty(shape, options);
    }

    return torch::from_blob(data, shape, options);
}

} // namespace tiny_llm
