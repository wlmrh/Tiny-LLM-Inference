#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
namespace tiny_llm {

/**
 * @enum DType
 * @brief Supported element data types for tensor operations.
 */
enum class DType { kFloat16, kFloat32, kInt32 };

/**
 * @class Tensor
 * @brief A lightweight tensor view over externally managed memory.
 *
 * @note CONTRACT:
 * - This object does not own the underlying memory buffer.
 * - It never performs allocation or deallocation on destruction.
 * - Multiple Tensor views may safely reference the same storage.
 */
class Tensor {
public:
    /**
     * @brief Default constructor. Creates an empty tensor with a null pointer.
     */
    Tensor() = default;

    /**
     * @brief Constructs a Tensor view from existing memory.
     * @param data Raw pointer to the external memory buffer.
     * @param shape Vector containing tensor dimensions.
     * @param dtype Scalar data type of tensor elements.
     */
    Tensor(void* data, std::vector<int64_t> shape, DType dtype)
        : data_ptr_(data), shape_(std::move(shape)), dtype_(dtype) {}

    /**
     * @brief Returns raw storage pointer.
     */
    void* data() const { return data_ptr_; }

    /**
     * @brief Returns tensor dimensions.
     */
    const std::vector<int64_t>& shape() const { return shape_; }

    /**
     * @brief Returns tensor element data type.
     */
    DType dtype() const { return dtype_; }

    /**
     * @brief Calculates total number of elements.
     */
    size_t numel() const;

private:
    /// Raw data pointer to external storage (non-owning).
    void* data_ptr_ = nullptr;
    /// Tensor dimensions in row-major logical order.
    std::vector<int64_t> shape_;
    /// Scalar element type for this tensor.
    DType dtype_ = DType::kFloat16;
};

} // namespace tiny_llm
