#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

namespace tiny_llm {

enum class DType { kFloat32, kFloat16, kInt8 };

class Tensor {
public:
    // Constructor: requires external pointer to GPU memory
    // Memory lifecycle is managed by external allocator (StackAllocator, BlockAllocator, etc.)
    // The Tensor class does NOT own the memory.
    Tensor(std::vector<int64_t> shape, DType dtype, void* gpu_ptr);
    ~Tensor() = default; // no memory cleanup needed

    // disable copy, enable move
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept = default;

    // information
    template<typename T>
    T* data() const { return static_cast<T*>(data_ptr_); } // return the address of the tensor
    const std::vector<int64_t>& shape() const { return shape_; }
    size_t numel() const;
    size_t size_in_bytes() const;
    DType dtype() const { return dtype_; }

private:
    void* data_ptr_ = nullptr;              // address of the tensor in GPU (managed by allocator)
    std::vector<int64_t> shape_;            // shape of the tensor
    DType dtype_;                           // data type of the elements
};

} // namespace tiny_llm
