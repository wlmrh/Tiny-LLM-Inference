#pragma once
#include <cstdint>
#include <vector>

namespace tiny_llm {

// Supported element data types.
enum class DType { kFloat16, kFloat32, kInt32 };

// Lightweight tensor view over externally managed memory.
class Tensor {
public:
    Tensor() = default;
    Tensor(void* data, std::vector<int64_t> shape, DType dtype)
        : data_ptr_(data), shape_(std::move(shape)), dtype_(dtype) {}

    // Returns raw storage pointer.
    void* data() const { return data_ptr_; }
    // Returns tensor dimensions.
    const std::vector<int64_t>& shape() const { return shape_; }
    // Returns tensor element type.
    DType dtype() const { return dtype_; }

    // Returns total element count (product of shape dimensions).
    size_t numel() const;

private:
    // Raw data pointer (typically GPU memory in this project).
    void* data_ptr_ = nullptr;
    // Tensor dimensions in row-major logical order.
    std::vector<int64_t> shape_;
    // Scalar element type for this tensor.
    DType dtype_ = DType::kFloat16;
};

} // namespace tiny_llm