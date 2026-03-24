#pragma once
#include <cstdint>
#include <vector>

namespace tiny_llm {

/**
 * @enum DType
 * @brief Supported element data types for tensor operations.
 */
enum class DType { 
    kFloat16, ///< 16-bit floating point (Half precision)
    kFloat32, ///< 32-bit floating point (Single precision)
    kInt32    ///< 32-bit signed integer
};

/**
 * @class Tensor
 * @brief A lightweight tensor view over externally managed memory.
 * * @note CONTRACT: 
 * - This object does not own the underlying memory buffer.
 * - It does not perform allocation or deallocation (no `free`/`delete`) on destruction.
 * - This design allows multiple @ref Tensor views to point to the same buffer without double-free risks.
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
     * @param shape Vector containing the dimensions of the tensor.
     * @param dtype The scalar data type of the elements.
     */
    Tensor(void* data, std::vector<int64_t> shape, DType dtype)
        : data_ptr_(data), shape_(std::move(shape)), dtype_(dtype) {}

    /**
     * @brief Returns the raw storage pointer.
     * @return void* Pointer to the data buffer.
     */
    void* data() const { return data_ptr_; }

    /**
     * @brief Returns the tensor dimensions.
     * @return A const reference to the shape vector.
     */
    const std::vector<int64_t>& shape() const { return shape_; }

    /**
     * @brief Returns the scalar element type of this tensor.
     * @return DType The data type (e.g., kFloat16).
     */
    DType dtype() const { return dtype_; }

    /**
     * @brief Calculates the total number of elements in the tensor.
     * @return The product of all dimensions in the shape.
     */
    size_t numel() const;

private:
    /// Raw data pointer to the external memory (non-owning).
    void* data_ptr_ = nullptr;
    
    /// Tensor dimensions in row-major logical order.
    std::vector<int64_t> shape_;
    
    /// Scalar element type for this tensor.
    DType dtype_ = DType::kFloat16;
};

} // namespace tiny_llm