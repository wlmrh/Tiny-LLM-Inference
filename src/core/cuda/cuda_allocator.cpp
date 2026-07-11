#include "tiny_llm/core/allocator.h"

#include "utils/cuda_utils.h"

#include <cstdlib>
#include <stdexcept>

namespace tiny_llm
{

StackAllocator::StackAllocator(size_t pool_size) : StackAllocator(pool_size, ParallelConfig::cpu()) {}

StackAllocator::StackAllocator(size_t pool_size, ParallelConfig parallel_config)
    : total_size_(pool_size), parallel_config_(parallel_config)
{
    parallel_config_.validate();
    if (parallel_config_.is_cuda())
    {
        CHECK_CUDA(cudaSetDevice(parallel_config_.device_id()));
        CHECK_CUDA(cudaMalloc(&base_ptr_, pool_size));
    }
    else
    {
        base_ptr_ = std::malloc(pool_size);
        if (base_ptr_ == nullptr)
        {
            throw std::runtime_error("StackAllocator: failed to allocate CPU workspace memory.");
        }
    }
}

StackAllocator::~StackAllocator()
{
    if (base_ptr_ != nullptr)
    {
        if (parallel_config_.is_cuda())
        {
            CHECK_CUDA(cudaSetDevice(parallel_config_.device_id()));
            CHECK_CUDA(cudaFree(base_ptr_));
        }
        else
        {
            std::free(base_ptr_);
        }
    }
}

void *StackAllocator::allocate(size_t bytes)
{
    if (offset_ + bytes > total_size_)
    {
        return nullptr;
    }
    void *ptr = static_cast<char *>(base_ptr_) + offset_;
    offset_ += bytes;
    return ptr;
}

void StackAllocator::reset()
{
    offset_ = 0;
}

} // namespace tiny_llm
