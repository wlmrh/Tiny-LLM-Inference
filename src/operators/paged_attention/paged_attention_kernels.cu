#if TINYLLM_ENABLE_CUDA
#include "utils/cuda_utils.h"

#include <cfloat>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace tiny_llm::ops::cuda
{

namespace
{

constexpr int kThreadsPerBlock = 256;
constexpr int kAttentionThreadsPerBlock = 1024;
constexpr int kFastAttentionMaxContextTokens = 2048;

__global__ void copy_f32_kernel(const float *src, float *dst, int64_t n)
{
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    if (idx >= n)
    {
        return;
    }
    dst[idx] = src[idx];
}

} // namespace

void launch_attention_paged_f32(const float *q, float *out, int64_t numel, cudaStream_t stream)
{
    if (numel <= 0)
    {
        return;
    }

    const int64_t blocks =
        (numel + static_cast<int64_t>(kThreadsPerBlock) - 1) / static_cast<int64_t>(kThreadsPerBlock);

    copy_f32_kernel<<<static_cast<unsigned int>(blocks), kThreadsPerBlock, 0, stream>>>(q, out, numel);
    CHECK_CUDA(cudaGetLastError());
}

__device__ int32_t block_id_for_position(const int32_t *block_tables, int64_t num_seqs, int64_t max_blocks_per_seq,
                                         int32_t block_size_tokens, int32_t layer_id, int32_t seq_index,
                                         int32_t position)
{
    const int32_t logical_block = position / block_size_tokens;
    if (logical_block < 0 || logical_block >= max_blocks_per_seq)
    {
        return -1;
    }
    const int64_t block_index = static_cast<int64_t>(layer_id) * num_seqs * max_blocks_per_seq +
                                static_cast<int64_t>(seq_index) * max_blocks_per_seq + logical_block;
    return block_tables[block_index];
}

template <typename T> __device__ float kv_to_float(T value)
{
    return static_cast<float>(value);
}

template <> __device__ float kv_to_float<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename T> __device__ T float_to_kv(float value)
{
    return static_cast<T>(value);
}

template <> __device__ __nv_bfloat16 float_to_kv<__nv_bfloat16>(float value)
{
    return __float2bfloat16(value);
}

template <typename KVType>
__global__ void write_paged_kv_cache_kernel(const float *k, const float *v, const int32_t *positions,
                                            const int32_t *seq_indices, const int32_t *block_tables,
                                            KVType *kv_pool_base, int64_t rows, int64_t num_seqs,
                                            int64_t max_blocks_per_seq, int64_t num_blocks, int64_t block_size_bytes,
                                            int32_t block_size_tokens, int32_t layer_id, int32_t kv_size)
{
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows)
    {
        return;
    }
    const int32_t seq_index = seq_indices[row];
    const int32_t position = positions[row];
    if (seq_index < 0 || seq_index >= num_seqs || position < 0)
    {
        return;
    }
    const int32_t block_id = block_id_for_position(block_tables, num_seqs, max_blocks_per_seq, block_size_tokens,
                                                   layer_id, seq_index, position);
    if (block_id < 0 || block_id >= num_blocks)
    {
        return;
    }

    const int32_t token_offset = position % block_size_tokens;
    KVType *block = reinterpret_cast<KVType *>(reinterpret_cast<char *>(kv_pool_base) +
                                               static_cast<int64_t>(block_id) * block_size_bytes);
    KVType *key_dst = block + static_cast<int64_t>(token_offset) * kv_size;
    KVType *value_dst =
        block + static_cast<int64_t>(block_size_tokens) * kv_size + static_cast<int64_t>(token_offset) * kv_size;
    const int64_t row_offset = row * kv_size;
    for (int32_t dim = static_cast<int32_t>(threadIdx.x); dim < kv_size; dim += static_cast<int32_t>(blockDim.x))
    {
        key_dst[dim] = float_to_kv<KVType>(k[row_offset + dim]);
        value_dst[dim] = float_to_kv<KVType>(v[row_offset + dim]);
    }
}

template <typename KVType>
__global__ void paged_attention_kernel(const float *q, float *out, const int32_t *positions, const int32_t *seq_indices,
                                       const int32_t *context_lens, const int32_t *block_tables,
                                       const KVType *kv_pool_base, int64_t rows, int64_t num_seqs,
                                       int64_t max_blocks_per_seq, int64_t num_blocks, int64_t block_size_bytes,
                                       int32_t block_size_tokens, int32_t layer_id, int32_t num_attention_heads,
                                       int32_t num_key_value_heads, int32_t head_dim)
{
    extern __shared__ float shared[];
    float *reduce = shared;
    float *aux = shared + blockDim.x;

    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int32_t q_head = static_cast<int32_t>(blockIdx.y);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    if (row >= rows || q_head >= num_attention_heads)
    {
        return;
    }

    const int32_t kv_size = num_key_value_heads * head_dim;
    const int32_t group_size = num_attention_heads / num_key_value_heads;
    const int32_t kv_head = q_head / group_size;
    const int32_t seq_index = seq_indices[row];
    const int32_t position = positions[row];
    if (seq_index < 0 || seq_index >= num_seqs || position < 0 || position >= context_lens[seq_index])
    {
        return;
    }

    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int64_t q_row_offset = row * static_cast<int64_t>(num_attention_heads) * head_dim;
    const int64_t q_head_offset = q_row_offset + static_cast<int64_t>(q_head) * head_dim;
    const int32_t context_tokens = position + 1;
    if (context_tokens <= kFastAttentionMaxContextTokens && head_dim <= static_cast<int32_t>(blockDim.x))
    {
        float local_score = -FLT_MAX;
        for (int32_t src_pos = tid; src_pos < context_tokens; src_pos += static_cast<int32_t>(blockDim.x))
        {
            const int32_t block_id = block_id_for_position(block_tables, num_seqs, max_blocks_per_seq,
                                                           block_size_tokens, layer_id, seq_index, src_pos);
            if (block_id >= 0 && block_id < num_blocks)
            {
                const int32_t src_offset = src_pos % block_size_tokens;
                const KVType *key_base =
                    reinterpret_cast<const KVType *>(reinterpret_cast<const char *>(kv_pool_base) +
                                                     static_cast<int64_t>(block_id) * block_size_bytes) +
                    static_cast<int64_t>(src_offset) * kv_size + static_cast<int64_t>(kv_head) * head_dim;
                float score = 0.0f;
                for (int32_t dim = 0; dim < head_dim; ++dim)
                {
                    score += q[q_head_offset + dim] * kv_to_float<KVType>(key_base[dim]);
                }
                const float scaled_score = score * scale;
                aux[src_pos] = scaled_score;
                local_score = fmaxf(local_score, scaled_score);
            }
        }
        reduce[tid] = local_score;
        __syncthreads();
        for (int32_t stride = static_cast<int32_t>(blockDim.x) / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
            }
            __syncthreads();
        }
        const float max_score = reduce[0];

        float local_sum = 0.0f;
        for (int32_t src_pos = tid; src_pos < context_tokens; src_pos += static_cast<int32_t>(blockDim.x))
        {
            local_sum += expf(aux[src_pos] - max_score);
        }
        reduce[tid] = local_sum;
        __syncthreads();
        for (int32_t stride = static_cast<int32_t>(blockDim.x) / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        const float score_sum = reduce[0];
        if (score_sum <= 0.0f)
        {
            return;
        }

        if (tid < head_dim)
        {
            float accum = 0.0f;
            for (int32_t src_pos = 0; src_pos < context_tokens; ++src_pos)
            {
                const int32_t block_id = block_id_for_position(block_tables, num_seqs, max_blocks_per_seq,
                                                               block_size_tokens, layer_id, seq_index, src_pos);
                if (block_id < 0 || block_id >= num_blocks)
                {
                    continue;
                }
                const int32_t src_offset = src_pos % block_size_tokens;
                const KVType *value_base =
                    reinterpret_cast<const KVType *>(reinterpret_cast<const char *>(kv_pool_base) +
                                                     static_cast<int64_t>(block_id) * block_size_bytes) +
                    static_cast<int64_t>(block_size_tokens) * kv_size + static_cast<int64_t>(src_offset) * kv_size +
                    static_cast<int64_t>(kv_head) * head_dim;
                accum += expf(aux[src_pos] - max_score) * kv_to_float<KVType>(value_base[tid]);
            }
            out[q_head_offset + tid] = accum / score_sum;
        }
        return;
    }

    float local_max = -FLT_MAX;
    for (int32_t src_pos = tid; src_pos <= position; src_pos += static_cast<int32_t>(blockDim.x))
    {
        const int32_t block_id = block_id_for_position(block_tables, num_seqs, max_blocks_per_seq, block_size_tokens,
                                                       layer_id, seq_index, src_pos);
        if (block_id < 0 || block_id >= num_blocks)
        {
            continue;
        }
        const int32_t src_offset = src_pos % block_size_tokens;
        const KVType *key_base = reinterpret_cast<const KVType *>(reinterpret_cast<const char *>(kv_pool_base) +
                                                                  static_cast<int64_t>(block_id) * block_size_bytes) +
                                 static_cast<int64_t>(src_offset) * kv_size + static_cast<int64_t>(kv_head) * head_dim;
        float score = 0.0f;
        for (int32_t dim = 0; dim < head_dim; ++dim)
        {
            score += q[q_head_offset + dim] * kv_to_float<KVType>(key_base[dim]);
        }
        local_max = fmaxf(local_max, score * scale);
    }

    reduce[tid] = local_max;
    __syncthreads();
    for (int32_t stride = static_cast<int32_t>(blockDim.x) / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];

    for (int32_t dim = tid; dim < head_dim; dim += static_cast<int32_t>(blockDim.x))
    {
        aux[dim] = 0.0f;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int32_t src_pos = tid; src_pos <= position; src_pos += static_cast<int32_t>(blockDim.x))
    {
        const int32_t block_id = block_id_for_position(block_tables, num_seqs, max_blocks_per_seq, block_size_tokens,
                                                       layer_id, seq_index, src_pos);
        if (block_id < 0 || block_id >= num_blocks)
        {
            continue;
        }
        const int32_t src_offset = src_pos % block_size_tokens;
        const KVType *block = reinterpret_cast<const KVType *>(reinterpret_cast<const char *>(kv_pool_base) +
                                                               static_cast<int64_t>(block_id) * block_size_bytes);
        const KVType *key_base =
            block + static_cast<int64_t>(src_offset) * kv_size + static_cast<int64_t>(kv_head) * head_dim;
        const KVType *value_base = block + static_cast<int64_t>(block_size_tokens) * kv_size +
                                   static_cast<int64_t>(src_offset) * kv_size +
                                   static_cast<int64_t>(kv_head) * head_dim;

        float score = 0.0f;
        for (int32_t dim = 0; dim < head_dim; ++dim)
        {
            score += q[q_head_offset + dim] * kv_to_float<KVType>(key_base[dim]);
        }
        const float weight = expf(score * scale - max_score);
        local_sum += weight;
        for (int32_t dim = 0; dim < head_dim; ++dim)
        {
            atomicAdd(&aux[dim], weight * kv_to_float<KVType>(value_base[dim]));
        }
    }

    reduce[tid] = local_sum;
    __syncthreads();
    for (int32_t stride = static_cast<int32_t>(blockDim.x) / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            reduce[tid] += reduce[tid + stride];
        }
        __syncthreads();
    }
    const float score_sum = reduce[0];
    if (score_sum <= 0.0f)
    {
        return;
    }

    for (int32_t dim = tid; dim < head_dim; dim += static_cast<int32_t>(blockDim.x))
    {
        out[q_head_offset + dim] = aux[dim] / score_sum;
    }
}

void launch_write_paged_kv_cache_f32(const float *k, const float *v, const int32_t *positions,
                                     const int32_t *seq_indices, const int32_t *block_tables, float *kv_pool_base,
                                     int64_t rows, int64_t num_seqs, int64_t max_blocks_per_seq, int64_t num_blocks,
                                     int64_t block_size_bytes, int32_t block_size_tokens, int32_t layer_id,
                                     int32_t kv_size, cudaStream_t stream);

void launch_paged_attention_query_f32(const float *q, float *out, const int32_t *positions,
                                      const int32_t *seq_indices, const int32_t *context_lens,
                                      const int32_t *block_tables, const float *kv_pool_base, int64_t rows,
                                      int64_t num_seqs, int64_t max_blocks_per_seq, int64_t num_blocks,
                                      int64_t block_size_bytes, int32_t block_size_tokens, int32_t layer_id,
                                      int32_t num_attention_heads, int32_t num_key_value_heads, int32_t head_dim,
                                      cudaStream_t stream)
{
    if (rows <= 0 || num_attention_heads <= 0 || num_key_value_heads <= 0 || head_dim <= 0)
    {
        return;
    }
    const dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(num_attention_heads));
    const size_t aux_floats = static_cast<size_t>(head_dim) > static_cast<size_t>(kFastAttentionMaxContextTokens)
                                  ? static_cast<size_t>(head_dim)
                                  : static_cast<size_t>(kFastAttentionMaxContextTokens);
    const size_t shared_bytes = (static_cast<size_t>(kAttentionThreadsPerBlock) + aux_floats) * sizeof(float);
    paged_attention_kernel<float><<<grid, kAttentionThreadsPerBlock, shared_bytes, stream>>>(
        q, out, positions, seq_indices, context_lens, block_tables, kv_pool_base, rows, num_seqs, max_blocks_per_seq,
        num_blocks, block_size_bytes, block_size_tokens, layer_id, num_attention_heads, num_key_value_heads, head_dim);
    CHECK_CUDA(cudaGetLastError());
}

void launch_paged_attention_f32(const float *q, const float *k, const float *v, float *out, const int32_t *positions,
                                const int32_t *seq_indices, const int32_t *context_lens, const int32_t *block_tables,
                                float *kv_pool_base, int64_t rows, int64_t num_seqs, int64_t max_blocks_per_seq,
                                int64_t num_blocks, int64_t block_size_bytes, int32_t block_size_tokens,
                                int32_t layer_id, int32_t num_attention_heads, int32_t num_key_value_heads,
                                int32_t head_dim, cudaStream_t stream)
{
    if (rows <= 0 || num_attention_heads <= 0 || num_key_value_heads <= 0 || head_dim <= 0)
    {
        return;
    }
    const int32_t kv_size = num_key_value_heads * head_dim;
    launch_write_paged_kv_cache_f32(k, v, positions, seq_indices, block_tables, kv_pool_base, rows, num_seqs,
                                    max_blocks_per_seq, num_blocks, block_size_bytes, block_size_tokens, layer_id,
                                    kv_size, stream);
    launch_paged_attention_query_f32(q, out, positions, seq_indices, context_lens, block_tables, kv_pool_base, rows,
                                     num_seqs, max_blocks_per_seq, num_blocks, block_size_bytes, block_size_tokens,
                                     layer_id, num_attention_heads, num_key_value_heads, head_dim, stream);
}

void launch_write_paged_kv_cache_f32(const float *k, const float *v, const int32_t *positions,
                                     const int32_t *seq_indices, const int32_t *block_tables, float *kv_pool_base,
                                     int64_t rows, int64_t num_seqs, int64_t max_blocks_per_seq, int64_t num_blocks,
                                     int64_t block_size_bytes, int32_t block_size_tokens, int32_t layer_id,
                                     int32_t kv_size, cudaStream_t stream)
{
    if (rows <= 0 || kv_size <= 0)
    {
        return;
    }
    write_paged_kv_cache_kernel<float><<<static_cast<unsigned int>(rows), kThreadsPerBlock, 0, stream>>>(
        k, v, positions, seq_indices, block_tables, kv_pool_base, rows, num_seqs, max_blocks_per_seq, num_blocks,
        block_size_bytes, block_size_tokens, layer_id, kv_size);
    CHECK_CUDA(cudaGetLastError());
}

void launch_paged_attention_bf16_kv(const float *q, const float *k, const float *v, float *out,
                                    const int32_t *positions, const int32_t *seq_indices, const int32_t *context_lens,
                                    const int32_t *block_tables, void *kv_pool_base, int64_t rows, int64_t num_seqs,
                                    int64_t max_blocks_per_seq, int64_t num_blocks, int64_t block_size_bytes,
                                    int32_t block_size_tokens, int32_t layer_id, int32_t num_attention_heads,
                                    int32_t num_key_value_heads, int32_t head_dim, cudaStream_t stream)
{
    if (rows <= 0 || num_attention_heads <= 0 || num_key_value_heads <= 0 || head_dim <= 0)
    {
        return;
    }
    const int32_t kv_size = num_key_value_heads * head_dim;
    auto *pool = static_cast<__nv_bfloat16 *>(kv_pool_base);
    write_paged_kv_cache_kernel<__nv_bfloat16><<<static_cast<unsigned int>(rows), kThreadsPerBlock, 0, stream>>>(
        k, v, positions, seq_indices, block_tables, pool, rows, num_seqs, max_blocks_per_seq, num_blocks,
        block_size_bytes, block_size_tokens, layer_id, kv_size);
    CHECK_CUDA(cudaGetLastError());

    const dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(num_attention_heads));
    const size_t aux_floats = static_cast<size_t>(head_dim) > static_cast<size_t>(kFastAttentionMaxContextTokens)
                                  ? static_cast<size_t>(head_dim)
                                  : static_cast<size_t>(kFastAttentionMaxContextTokens);
    const size_t shared_bytes = (static_cast<size_t>(kAttentionThreadsPerBlock) + aux_floats) * sizeof(float);
    paged_attention_kernel<__nv_bfloat16><<<grid, kAttentionThreadsPerBlock, shared_bytes, stream>>>(
        q, out, positions, seq_indices, context_lens, block_tables, pool, rows, num_seqs, max_blocks_per_seq,
        num_blocks, block_size_bytes, block_size_tokens, layer_id, num_attention_heads, num_key_value_heads, head_dim);
    CHECK_CUDA(cudaGetLastError());
}

} // namespace tiny_llm::ops::cuda
#endif
