#include "tiny_llm/core/allocator.h"

#include <cstdlib>
#include <stdexcept>

namespace tiny_llm
{

StackAllocator::StackAllocator(size_t pool_size) : StackAllocator(pool_size, ParallelConfig::cpu()) {}

StackAllocator::StackAllocator(size_t pool_size, ParallelConfig parallel_config)
    : total_size_(pool_size), parallel_config_(parallel_config)
{
    parallel_config_.validate();
    if (!parallel_config_.is_cpu())
    {
        throw std::runtime_error("StackAllocator: CPU build only supports CPU workspace memory.");
    }
    base_ptr_ = std::malloc(pool_size);
}

StackAllocator::~StackAllocator()
{
    if (base_ptr_ != nullptr)
    {
        std::free(base_ptr_);
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
