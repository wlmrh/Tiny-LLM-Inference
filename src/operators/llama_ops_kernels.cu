#if TINYLLM_ENABLE_CUDA
#include "utils/cuda_utils.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace tiny_llm::ops::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void apply_rope_cached_f32_kernel(const int32_t* positions,
                                             float* q,
                                             float* k,
                                             const float* cos_cache,
                                             const float* sin_cache,
                                             int64_t rows,
                                             int32_t num_attention_heads,
                                             int32_t num_key_value_heads,
                                             int32_t head_dim,
                                             int64_t cache_rows,
                                             int64_t q_stride,
                                             int64_t k_stride)
{
    const int32_t rotary_half = head_dim / 2;
    const int64_t q_items = rows
        * static_cast<int64_t>(num_attention_heads)
        * static_cast<int64_t>(rotary_half);
    const int64_t k_items = rows
        * static_cast<int64_t>(num_key_value_heads)
        * static_cast<int64_t>(rotary_half);
    const int64_t total = q_items + k_items;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x)
        + static_cast<int64_t>(threadIdx.x);
    if (idx >= total)
    {
        return;
    }

    const bool is_q = idx < q_items;
    const int64_t local = is_q ? idx : idx - q_items;
    const int32_t heads = is_q ? num_attention_heads : num_key_value_heads;
    const int64_t per_row = static_cast<int64_t>(heads) * static_cast<int64_t>(rotary_half);
    const int64_t row = local / per_row;
    const int64_t rem = local - row * per_row;
    const int32_t head = static_cast<int32_t>(rem / rotary_half);
    const int32_t dim = static_cast<int32_t>(rem - static_cast<int64_t>(head) * rotary_half);
    if (row >= rows)
    {
        return;
    }

    const int32_t position = positions[row];
    if (position < 0 || static_cast<int64_t>(position) >= cache_rows)
    {
        return;
    }

    const float c = cos_cache[static_cast<int64_t>(position) * rotary_half + dim];
    const float s = sin_cache[static_cast<int64_t>(position) * rotary_half + dim];
    float* base = is_q ? q : k;
    const int64_t stride = is_q ? q_stride : k_stride;
    const int64_t offset = row * stride + static_cast<int64_t>(head) * head_dim + dim;
    const int64_t offset_pair = offset + rotary_half;
    const float v0 = base[offset];
    const float v1 = base[offset_pair];
    base[offset] = v0 * c - v1 * s;
    base[offset_pair] = v1 * c + v0 * s;
}

__global__ void silu_multiply_f32_kernel(const float* gate,
                                         const float* up,
                                         float* out,
                                         int64_t numel)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x)
        + static_cast<int64_t>(threadIdx.x);
    if (idx >= numel)
    {
        return;
    }
    const float gate_value = gate[idx];
    out[idx] = (gate_value / (1.0f + expf(-gate_value))) * up[idx];
}

} // namespace

void launch_apply_rope_cached_f32(const int32_t* positions,
                                  float* q,
                                  float* k,
                                  const float* cos_cache,
                                  const float* sin_cache,
                                  int64_t rows,
                                  int32_t num_attention_heads,
                                  int32_t num_key_value_heads,
                                  int32_t head_dim,
                                  int64_t cache_rows,
                                  int64_t q_stride,
                                  int64_t k_stride,
                                  cudaStream_t stream)
{
    if (rows <= 0 || head_dim <= 0 || head_dim % 2 != 0)
    {
        return;
    }
    const int64_t rotary_half = head_dim / 2;
    const int64_t total = rows
        * static_cast<int64_t>(num_attention_heads + num_key_value_heads)
        * rotary_half;
    if (total <= 0)
    {
        return;
    }
    const int64_t blocks = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;
    apply_rope_cached_f32_kernel<<<static_cast<unsigned int>(blocks), kThreadsPerBlock, 0, stream>>>(
        positions,
        q,
        k,
        cos_cache,
        sin_cache,
        rows,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        cache_rows,
        q_stride,
        k_stride);
    CHECK_CUDA(cudaGetLastError());
}

void launch_silu_multiply_f32(const float* gate,
                              const float* up,
                              float* out,
                              int64_t numel,
                              cudaStream_t stream)
{
    if (numel <= 0)
    {
        return;
    }
    const int64_t blocks = (numel + kThreadsPerBlock - 1) / kThreadsPerBlock;
    silu_multiply_f32_kernel<<<static_cast<unsigned int>(blocks), kThreadsPerBlock, 0, stream>>>(
        gate,
        up,
        out,
        numel);
    CHECK_CUDA(cudaGetLastError());
}

} // namespace tiny_llm::ops::cuda
#endif
