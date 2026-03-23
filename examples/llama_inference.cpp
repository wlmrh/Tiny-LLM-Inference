#include <iostream>

#include "tiny_llm/core/allocator.h"
#include "tiny_llm/core/tensor.h"

int main() {
    tiny_llm::StackAllocator allocator(1024 * 1024);
    const tiny_llm::Tensor logits = allocator.make_tensor({1, 32000}, tiny_llm::DType::kFloat32);

    std::cout << "Tiny-LLM inference example bootstrap" << std::endl;
    std::cout << "logits elements: " << logits.numel() << std::endl;
    return 0;
}
