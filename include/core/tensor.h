#pragma once
#include <cstdint>
#include <vector>

namespace tiny_llm {

enum class DType { kFloat16, kFloat32, kInt32 };

class Tensor {
public:
    Tensor() = default;
    Tensor(void* data, std::vector<int64_t> shape, DType dtype)
        : data_(data), shape_(std::move(shape)), dtype_(dtype) {}

    void* data() const { return data_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }

    size_t numel() const;

private:
    void* data_ = nullptr;                 // device pointer
    std::vector<int64_t> shape_;
    DType dtype_ = DType::kFloat16;
};

} // namespace tiny_llm