#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tiny_llm {

class Tensor;
enum class DType;

class StackAllocator {
public:
    explicit StackAllocator(size_t bytes);
    ~StackAllocator();

    void reset();                 // called at begin_step
    void* allocate(size_t bytes); // bump pointer

    Tensor make_tensor(std::vector<int64_t> shape, DType dtype);

private:
    void* base_ = nullptr;  // device pointer
    size_t cap_ = 0;
    size_t off_ = 0;
};

class BlockAllocator {
public:
    BlockAllocator(void* pool, size_t num_blocks, size_t bytes_per_block);

    int32_t alloc_block();            // returns block_id or -1
    void free_block(int32_t block_id);
    void* block_ptr(int32_t block_id) const;

private:
    void* pool_ = nullptr;
    size_t n_ = 0;
    size_t bsz_ = 0;
    std::vector<int32_t> free_;
};

} // namespace tiny_llm