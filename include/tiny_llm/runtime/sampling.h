#pragma once

#include <cstdint>
#include <stdexcept>

namespace tiny_llm {

/**
 * @brief Greedy sampler that returns the argmax token ID.
 */
inline int32_t sample_argmax(const float* logits, int32_t vocab)
{
    if (logits == nullptr)
    {
        throw std::runtime_error("sample_argmax: logits pointer must be non-null.");
    }
    if (vocab <= 0)
    {
        throw std::runtime_error("sample_argmax: vocab must be positive.");
    }

    int32_t best_id = 0;
    float best_logit = logits[0];

    //< find the most likely vocabulary
    for (int32_t i = 1; i < vocab; ++i)
    {
        if (logits[i] > best_logit)
        {
            best_logit = logits[i];
            best_id = i;
        }
    }
    return best_id;
}

} // namespace tiny_llm
