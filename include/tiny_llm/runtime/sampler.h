#pragma once

#include <cstdint>
#include <vector>

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/processors.h"

namespace tiny_llm {

std::vector<int32_t> sample_greedy_rows(const Tensor& logits,
                                        const std::vector<int32_t>& sample_rows,
                                        int32_t vocab_size,
                                        const std::vector<std::vector<int32_t>>* token_histories = nullptr,
                                        const std::vector<SamplingParams>* sampling_params = nullptr,
                                        const std::vector<uint64_t>* request_ids = nullptr);

} // namespace tiny_llm
