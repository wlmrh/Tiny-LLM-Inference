#pragma once

#include <cuda_runtime.h>
#include <memory>

namespace tiny_llm {

class StackAllocator;
class BlockAllocator;

// Runtime context carries allocators and CUDA stream
// Passed through operator layers for resource management
struct OpContext {
    cudaStream_t stream = nullptr;
    StackAllocator* scratch_pad = nullptr;    // For temporary tensors
    BlockAllocator* block_alloc = nullptr;    // For KV cache blocks
};

} // namespace tiny_llm
