#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tiny_llm {

class Tensor;
enum class DType;

class StackAllocator {
public:
    explicit StackAllocator(size_t pool_size);
    ~StackAllocator();

    void reset();
    void* allocate(size_t bytes);
    Tensor make_tensor(std::vector<int64_t> shape, DType dtype);

private:
    void* base_ptr_ = nullptr;
    size_t total_size_ = 0;
    size_t offset_ = 0;
};

class BlockAllocator {
public:
    BlockAllocator(size_t num_blocks, size_t block_size_bytes, void* gpu_pool);

    int32_t allocate_block();
    void free_block(int32_t block_id);
    void* get_block_ptr(int32_t block_id) const;

private:
    void* gpu_pool_ = nullptr;
    size_t num_blocks_ = 0;
    size_t block_size_ = 0;
    std::vector<int32_t> free_list_;
};

} // namespace tiny_llm
