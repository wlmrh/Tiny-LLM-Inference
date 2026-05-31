#if TINYLLM_ENABLE_CUDA
#include "utils/cuda_utils.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace tiny_llm::cuda {

namespace {

constexpr int kSamplerThreads = 256;

__device__ bool better_argmax_candidate(float value, int32_t token, float best_value, int32_t best_token)
{
    return value > best_value || (value == best_value && token < best_token);
}

__global__ void mark_repetition_history_mask_kernel(const int64_t* history_tokens,
                                                    const int32_t* history_offsets,
                                                    uint8_t* history_mask,
                                                    int32_t sample_count,
                                                    int32_t vocab_size)
{
    const int32_t sample = static_cast<int32_t>(blockIdx.x);
    if (sample >= sample_count)
    {
        return;
    }

    const int32_t begin = history_offsets[sample];
    const int32_t end = history_offsets[sample + 1];
    uint8_t* row_mask = history_mask + static_cast<size_t>(sample) * static_cast<size_t>(vocab_size);
    for (int32_t index = begin + static_cast<int32_t>(threadIdx.x);
         index < end;
         index += static_cast<int32_t>(blockDim.x))
    {
        row_mask[history_tokens[index]] = 1;
    }
}

__global__ void argmax_with_repetition_penalty_kernel(const float* logits,
                                                      const int32_t* sample_rows,
                                                      const uint8_t* history_mask,
                                                      const float* penalties,
                                                      int32_t* sampled_tokens,
                                                      int32_t sample_count,
                                                      int32_t vocab_size)
{
    const int32_t sample = static_cast<int32_t>(blockIdx.x);
    if (sample >= sample_count)
    {
        return;
    }

    const int32_t row = sample_rows[sample];
    const float penalty = penalties[sample];
    const float* row_logits = logits + static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
    const uint8_t* row_mask = history_mask + static_cast<size_t>(sample) * static_cast<size_t>(vocab_size);

    float best_value = -std::numeric_limits<float>::infinity();
    int32_t best_token = vocab_size;
    for (int32_t token = static_cast<int32_t>(threadIdx.x);
         token < vocab_size;
         token += static_cast<int32_t>(blockDim.x))
    {
        float value = row_logits[token];
        if (row_mask[token] != 0 && penalty != 1.0f)
        {
            value = value > 0.0f ? value / penalty : value * penalty;
        }
        if (better_argmax_candidate(value, token, best_value, best_token))
        {
            best_value = value;
            best_token = token;
        }
    }

    __shared__ float shared_values[kSamplerThreads];
    __shared__ int32_t shared_tokens[kSamplerThreads];
    shared_values[threadIdx.x] = best_value;
    shared_tokens[threadIdx.x] = best_token;
    __syncthreads();

    for (int32_t stride = kSamplerThreads / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < static_cast<unsigned int>(stride))
        {
            const float other_value = shared_values[threadIdx.x + stride];
            const int32_t other_token = shared_tokens[threadIdx.x + stride];
            if (better_argmax_candidate(
                    other_value,
                    other_token,
                    shared_values[threadIdx.x],
                    shared_tokens[threadIdx.x]))
            {
                shared_values[threadIdx.x] = other_value;
                shared_tokens[threadIdx.x] = other_token;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        sampled_tokens[sample] = shared_tokens[0];
    }
}

} // namespace

void launch_mark_repetition_history_mask(const int64_t* history_tokens,
                                         const int32_t* history_offsets,
                                         uint8_t* history_mask,
                                         int32_t sample_count,
                                         int32_t vocab_size,
                                         cudaStream_t stream)
{
    if (sample_count <= 0 || vocab_size <= 0)
    {
        return;
    }

    mark_repetition_history_mask_kernel<<<sample_count, kSamplerThreads, 0, stream>>>(
        history_tokens,
        history_offsets,
        history_mask,
        sample_count,
        vocab_size);
    CHECK_CUDA(cudaGetLastError());
}

void launch_argmax_repetition_penalty_f32(const float* logits,
                                          const int32_t* sample_rows,
                                          const uint8_t* history_mask,
                                          const float* penalties,
                                          int32_t* sampled_tokens,
                                          int32_t sample_count,
                                          int32_t vocab_size,
                                          cudaStream_t stream)
{
    if (sample_count <= 0 || vocab_size <= 0)
    {
        return;
    }

    argmax_with_repetition_penalty_kernel<<<sample_count, kSamplerThreads, 0, stream>>>(
        logits,
        sample_rows,
        history_mask,
        penalties,
        sampled_tokens,
        sample_count,
        vocab_size);
    CHECK_CUDA(cudaGetLastError());
}

} // namespace tiny_llm::cuda
#endif
