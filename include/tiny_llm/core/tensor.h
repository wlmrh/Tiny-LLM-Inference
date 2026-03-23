#pragma once

#include <cstdint>
#include <vector>

namespace tiny_llm {

enum class DType { kFloat16, kFloat32, kInt32 };

// Lightweight tensor view over externally managed memory.
class Tensor {
public:
    Tensor() = default;
    Tensor(void* data, std::vector<int64_t> shape, DType dtype)
        : data_ptr_(data), shape_(std::move(shape)), dtype_(dtype) {}

    void* data() const { return data_ptr_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    size_t numel() const;

private:
    void* data_ptr_ = nullptr;
    std::vector<int64_t> shape_;
    DType dtype_ = DType::kFloat16;
};

} // namespace tiny_llm
