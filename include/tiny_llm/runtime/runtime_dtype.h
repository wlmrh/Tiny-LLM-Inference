#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace tiny_llm
{

enum class RuntimeDType
{
    kFloat32 = 0,
    kBFloat16 = 1,
};

inline const char *runtime_dtype_name(RuntimeDType dtype)
{
    switch (dtype)
    {
    case RuntimeDType::kFloat32:
        return "float32";
    case RuntimeDType::kBFloat16:
        return "bfloat16";
    }
    throw std::runtime_error("runtime_dtype_name: unsupported runtime dtype.");
}

inline size_t runtime_dtype_size(RuntimeDType dtype)
{
    switch (dtype)
    {
    case RuntimeDType::kFloat32:
        return sizeof(float);
    case RuntimeDType::kBFloat16:
        return sizeof(uint16_t);
    }
    throw std::runtime_error("runtime_dtype_size: unsupported runtime dtype.");
}

inline RuntimeDType parse_runtime_dtype(const std::string &text)
{
    if (text == "float32" || text == "fp32")
    {
        return RuntimeDType::kFloat32;
    }
    if (text == "bfloat16" || text == "bf16")
    {
        return RuntimeDType::kBFloat16;
    }
    throw std::runtime_error("runtime dtype must be float32/fp32 or bfloat16/bf16.");
}

} // namespace tiny_llm
