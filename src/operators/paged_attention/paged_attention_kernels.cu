#if TINYLLM_ENABLE_CUDA
#include "utils/cuda_utils.h"

#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

namespace tiny_llm::ops::cuda {

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void copy_f32_kernel(const float* src, float* dst, int64_t n)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x)
        + static_cast<int64_t>(threadIdx.x);
    if (idx >= n)
    {
        return;
    }
    dst[idx] = src[idx];
}

} // namespace

void launch_attention_paged_f32(const float* q, float* out, int64_t numel, cudaStream_t stream)
{
    if (numel <= 0)
    {
        return;
    }

    const int64_t blocks = (numel + static_cast<int64_t>(kThreadsPerBlock) - 1)
        / static_cast<int64_t>(kThreadsPerBlock);

    copy_f32_kernel<<<static_cast<unsigned int>(blocks), kThreadsPerBlock, 0, stream>>>(q, out, numel);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void paged_attention_decode_f32_kernel(const float* q,
                                                  const float* k,
                                                  const float* v,
                                                  float* out,
                                                  const int32_t* positions,
                                                  const int32_t* seq_indices,
                                                  const int32_t* context_lens,
                                                  const int32_t* block_tables,
                                                  float* kv_pool_base,
                                                  int64_t rows,
                                                  int64_t num_seqs,
                                                  int64_t max_blocks_per_seq,
                                                  int64_t block_size_bytes,
                                                  int32_t block_size_tokens,
                                                  int32_t layer_id,
                                                  int32_t num_attention_heads,
                                                  int32_t num_key_value_heads,
                                                  int32_t head_dim)
{
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int32_t q_head = static_cast<int32_t>(blockIdx.y);
    if (row >= rows || q_head >= num_attention_heads || threadIdx.x != 0)
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

    const int32_t current_logical_block = position / block_size_tokens;
    if (current_logical_block < 0 || current_logical_block >= max_blocks_per_seq)
    {
        return;
    }
    const int64_t current_block_index =
        static_cast<int64_t>(layer_id) * num_seqs * max_blocks_per_seq
        + static_cast<int64_t>(seq_index) * max_blocks_per_seq
        + current_logical_block;
    const int32_t current_block_id = block_tables[current_block_index];
    if (current_block_id < 0)
    {
        return;
    }
    const int32_t token_offset = position % block_size_tokens;
    float* current_block = reinterpret_cast<float*>(reinterpret_cast<char*>(kv_pool_base)
        + static_cast<int64_t>(current_block_id) * block_size_bytes);
    float* key_dst = current_block + static_cast<int64_t>(token_offset) * kv_size;
    float* value_dst = current_block
        + static_cast<int64_t>(block_size_tokens) * kv_size
        + static_cast<int64_t>(token_offset) * kv_size;
    const int64_t kv_row_offset = row * kv_size;
    for (int32_t dim = 0; dim < kv_size; ++dim)
    {
        key_dst[dim] = k[kv_row_offset + dim];
        value_dst[dim] = v[kv_row_offset + dim];
    }

    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int64_t q_row_offset = row * static_cast<int64_t>(num_attention_heads) * head_dim;
    float max_score = -FLT_MAX;
    for (int32_t src_pos = 0; src_pos <= position; ++src_pos)
    {
        const int32_t logical_block = src_pos / block_size_tokens;
        if (logical_block < 0 || logical_block >= max_blocks_per_seq)
        {
            return;
        }
        const int64_t block_index =
            static_cast<int64_t>(layer_id) * num_seqs * max_blocks_per_seq
            + static_cast<int64_t>(seq_index) * max_blocks_per_seq
            + logical_block;
        const int32_t block_id = block_tables[block_index];
        if (block_id < 0)
        {
            return;
        }
        const int32_t src_offset = src_pos % block_size_tokens;
        const float* key_base = reinterpret_cast<const float*>(reinterpret_cast<const char*>(kv_pool_base)
            + static_cast<int64_t>(block_id) * block_size_bytes)
            + static_cast<int64_t>(src_offset) * kv_size
            + static_cast<int64_t>(kv_head) * head_dim;
        float score = 0.0f;
        for (int32_t dim = 0; dim < head_dim; ++dim)
        {
            score += q[q_row_offset + static_cast<int64_t>(q_head) * head_dim + dim] * key_base[dim];
        }
        score *= scale;
        max_score = fmaxf(max_score, score);
    }

    float score_sum = 0.0f;
    for (int32_t dim = 0; dim < head_dim; ++dim)
    {
        out[q_row_offset + static_cast<int64_t>(q_head) * head_dim + dim] = 0.0f;
    }
    for (int32_t src_pos = 0; src_pos <= position; ++src_pos)
    {
        const int32_t logical_block = src_pos / block_size_tokens;
        if (logical_block < 0 || logical_block >= max_blocks_per_seq)
        {
            return;
        }
        const int64_t block_index =
            static_cast<int64_t>(layer_id) * num_seqs * max_blocks_per_seq
            + static_cast<int64_t>(seq_index) * max_blocks_per_seq
            + logical_block;
        const int32_t block_id = block_tables[block_index];
        if (block_id < 0)
        {
            return;
        }
        const int32_t src_offset = src_pos % block_size_tokens;
        const float* block = reinterpret_cast<const float*>(reinterpret_cast<const char*>(kv_pool_base)
            + static_cast<int64_t>(block_id) * block_size_bytes);
        const float* key_base = block
            + static_cast<int64_t>(src_offset) * kv_size
            + static_cast<int64_t>(kv_head) * head_dim;
        const float* value_base = block
            + static_cast<int64_t>(block_size_tokens) * kv_size
            + static_cast<int64_t>(src_offset) * kv_size
            + static_cast<int64_t>(kv_head) * head_dim;

        float score = 0.0f;
        for (int32_t dim = 0; dim < head_dim; ++dim)
        {
            score += q[q_row_offset + static_cast<int64_t>(q_head) * head_dim + dim] * key_base[dim];
        }
        const float weight = expf(score * scale - max_score);
        score_sum += weight;
        for (int32_t dim = 0; dim < head_dim; ++dim)
        {
            out[q_row_offset + static_cast<int64_t>(q_head) * head_dim + dim] += weight * value_base[dim];
        }
    }
    if (score_sum <= 0.0f)
    {
        return;
    }
    for (int32_t dim = 0; dim < head_dim; ++dim)
    {
        out[q_row_offset + static_cast<int64_t>(q_head) * head_dim + dim] /= score_sum;
    }
}

void launch_paged_attention_decode_f32(const float* q,
                                       const float* k,
                                       const float* v,
                                       float* out,
                                       const int32_t* positions,
                                       const int32_t* seq_indices,
                                       const int32_t* context_lens,
                                       const int32_t* block_tables,
                                       float* kv_pool_base,
                                       int64_t rows,
                                       int64_t num_seqs,
                                       int64_t max_blocks_per_seq,
                                       int64_t block_size_bytes,
                                       int32_t block_size_tokens,
                                       int32_t layer_id,
                                       int32_t num_attention_heads,
                                       int32_t num_key_value_heads,
                                       int32_t head_dim,
                                       cudaStream_t stream)
{
    if (rows <= 0 || num_attention_heads <= 0)
    {
        return;
    }
    const dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(num_attention_heads));
    paged_attention_decode_f32_kernel<<<grid, 1, 0, stream>>>(
        q,
        k,
        v,
        out,
        positions,
        seq_indices,
        context_lens,
        block_tables,
        kv_pool_base,
        rows,
        num_seqs,
        max_blocks_per_seq,
        block_size_bytes,
        block_size_tokens,
        layer_id,
        num_attention_heads,
        num_key_value_heads,
        head_dim);
    CHECK_CUDA(cudaGetLastError());
}

} // namespace tiny_llm::ops::cuda
#endif
