#pragma once

#include <cstdint>
#include <vector>

#include "tiny_llm/core/tensor.h"

namespace tiny_llm {

std::vector<int32_t> sample_greedy_rows(const Tensor& logits,
                                        const std::vector<int32_t>& sample_rows,
                                        int32_t vocab_size);

} // namespace tiny_llm
