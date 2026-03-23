#include <cassert>
#include <stdexcept>

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/operators/rmsnorm.h"

int main() {
    tiny_llm::Tensor x(nullptr, {8}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor w(nullptr, {8}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor y(nullptr, {8}, tiny_llm::DType::kFloat32);
    tiny_llm::ExecutionContext ctx(nullptr, nullptr, nullptr);

    bool threw = false;
    try {
        tiny_llm::ops::rmsnorm(x, w, y, ctx);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);
    return 0;
}
