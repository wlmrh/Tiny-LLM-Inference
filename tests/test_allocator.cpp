#include <cassert>

#include "tiny_llm/core/allocator.h"
#include "tiny_llm/core/tensor.h"

int main() {
    tiny_llm::StackAllocator allocator(1024);
    tiny_llm::Tensor t = allocator.make_tensor({8, 8}, tiny_llm::DType::kFloat16);

    assert(t.data() != nullptr);
    assert(t.numel() == 64);

    allocator.reset();
    return 0;
}
