#include "paged_attention_internal.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/kv_cache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tiny_llm {
namespace ops {

namespace {

void validate_cpu_tensor(const Tensor& tensor, const char* name)
{
    if (tensor.device().is_cuda())
    {
        throw std::runtime_error(std::string(name) + ": CUDA path reached CPU backend.");
    }
}

int32_t block_id_for(const LlamaAttentionParams& params,
                     const int32_t* block_ptr,
                     int64_t num_seqs,
                     int64_t max_blocks_per_seq,
                     int32_t seq_index,
                     int32_t position)
{
    if (seq_index < 0 || seq_index >= num_seqs)
    {
        throw std::runtime_error("llama_attention: seq index out of range.");
    }
    const int32_t logical_block = position / params.metadata->block_size_tokens;
    if (logical_block < 0 || logical_block >= max_blocks_per_seq)
    {
        throw std::runtime_error("llama_attention: logical block out of range.");
    }
    const int64_t index =
        static_cast<int64_t>(params.layer_id) * num_seqs * max_blocks_per_seq
        + static_cast<int64_t>(seq_index) * max_blocks_per_seq
        + logical_block;
    const int32_t block_id = block_ptr[index];
    if (block_id < 0)
    {
        throw std::runtime_error("llama_attention: missing physical KV block.");
    }
    return block_id;
}

float* k_block_ptr(KVCache& kv_cache, int32_t block_id)
{
    void* block = kv_cache.block_ptr(block_id);
    if (block == nullptr)
    {
        throw std::runtime_error("llama_attention: KV block pointer is null.");
    }
    return static_cast<float*>(block);
}

float* v_block_ptr(KVCache& kv_cache, int32_t block_id, int32_t block_size_tokens, int32_t kv_size)
{
    return k_block_ptr(kv_cache, block_id)
        + static_cast<size_t>(block_size_tokens) * static_cast<size_t>(kv_size);
}

} // namespace

void run_direct_attention_cpu(const LlamaAttentionParams& params)
{
    validate_cpu_tensor(*params.positions, "llama_attention");
    validate_cpu_tensor(*params.q, "llama_attention");
    validate_cpu_tensor(*params.k, "llama_attention");
    validate_cpu_tensor(*params.v, "llama_attention");
    validate_cpu_tensor(*params.out, "llama_attention");

    const int32_t* positions_ptr = static_cast<const int32_t*>(tensor_data(*params.positions));
    const float* q_ptr = static_cast<const float*>(tensor_data(*params.q));
    const float* k_ptr = static_cast<const float*>(tensor_data(*params.k));
    const float* v_ptr = static_cast<const float*>(tensor_data(*params.v));
    float* out_ptr = static_cast<float*>(tensor_data(*params.out));

    const int32_t q_hidden_size = attention_hidden_size(params.num_attention_heads, params.head_dim);
    const int32_t kv_size = kv_hidden_size(params.num_key_value_heads, params.head_dim);
    const int32_t group_size = params.num_attention_heads / params.num_key_value_heads;
    const int64_t rows = params.q->size(0);
    const float scale = 1.0f / std::sqrt(static_cast<float>(params.head_dim));

    std::vector<float> scores(static_cast<size_t>(rows), -std::numeric_limits<float>::infinity());
    for (int64_t row = 0; row < rows; ++row)
    {
        const size_t q_row_offset = static_cast<size_t>(row) * static_cast<size_t>(q_hidden_size);
        const int32_t target_position = positions_ptr[row];
        for (int32_t q_head = 0; q_head < params.num_attention_heads; ++q_head)
        {
            const int32_t kv_head = q_head / group_size;
            float max_score = -std::numeric_limits<float>::infinity();

            for (int64_t src = 0; src < rows; ++src)
            {
                if (src > row || positions_ptr[src] > target_position)
                {
                    scores[static_cast<size_t>(src)] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                const size_t k_row_offset = static_cast<size_t>(src) * static_cast<size_t>(kv_size);
                float score = 0.0f;
                for (int32_t dim = 0; dim < params.head_dim; ++dim)
                {
                    const size_t q_index = q_row_offset + static_cast<size_t>(q_head * params.head_dim + dim);
                    const size_t k_index = k_row_offset + static_cast<size_t>(kv_head * params.head_dim + dim);
                    score += q_ptr[q_index] * k_ptr[k_index];
                }
                score *= scale;
                scores[static_cast<size_t>(src)] = score;
                max_score = std::max(max_score, score);
            }

            float score_sum = 0.0f;
            for (int64_t src = 0; src < rows; ++src)
            {
                if (scores[static_cast<size_t>(src)] == -std::numeric_limits<float>::infinity())
                {
                    continue;
                }
                const float exp_score = std::exp(scores[static_cast<size_t>(src)] - max_score);
                scores[static_cast<size_t>(src)] = exp_score;
                score_sum += exp_score;
            }
            if (score_sum <= 0.0f)
            {
                throw std::runtime_error("llama_attention: no causal source tokens.");
            }

            for (int32_t dim = 0; dim < params.head_dim; ++dim)
            {
                float value = 0.0f;
                for (int64_t src = 0; src < rows; ++src)
                {
                    const float exp_score = scores[static_cast<size_t>(src)];
                    if (exp_score == -std::numeric_limits<float>::infinity())
                    {
                        continue;
                    }
                    const size_t v_row_offset = static_cast<size_t>(src) * static_cast<size_t>(kv_size);
                    const size_t v_index = v_row_offset + static_cast<size_t>(kv_head * params.head_dim + dim);
                    value += (exp_score / score_sum) * v_ptr[v_index];
                }
                const size_t out_index = q_row_offset + static_cast<size_t>(q_head * params.head_dim + dim);
                out_ptr[out_index] = value;
            }
        }
    }
}

void run_paged_attention_cpu(const LlamaAttentionParams& params)
{
    validate_cpu_tensor(*params.positions, "llama_attention");
    validate_cpu_tensor(*params.q, "llama_attention");
    validate_cpu_tensor(*params.k, "llama_attention");
    validate_cpu_tensor(*params.v, "llama_attention");
    validate_cpu_tensor(*params.out, "llama_attention");

    KVCache& kv_cache = *params.ctx->kv();
    const PagedAttentionRuntimeMetadata& metadata = *params.metadata;
    const Tensor& seq_indices = *metadata.seq_indices;
    const Tensor& context_lens = *metadata.context_lens;
    const Tensor& block_tables = *metadata.block_tables;

    const int32_t* positions_ptr = static_cast<const int32_t*>(tensor_data(*params.positions));
    const float* q_ptr = static_cast<const float*>(tensor_data(*params.q));
    const float* k_ptr = static_cast<const float*>(tensor_data(*params.k));
    const float* v_ptr = static_cast<const float*>(tensor_data(*params.v));
    float* out_ptr = static_cast<float*>(tensor_data(*params.out));
    const int32_t* seq_index_ptr = seq_indices.data_ptr<int32_t>();
    const int32_t* context_ptr = context_lens.data_ptr<int32_t>();
    const int32_t* block_ptr = block_tables.data_ptr<int32_t>();

    const std::vector<int64_t> block_shape = tensor_shape(block_tables);
    const int64_t num_seqs = block_shape[1];
    const int64_t max_blocks_per_seq = block_shape[2];
    const int32_t q_hidden_size = attention_hidden_size(params.num_attention_heads, params.head_dim);
    const int32_t kv_size = kv_hidden_size(params.num_key_value_heads, params.head_dim);
    const int32_t group_size = params.num_attention_heads / params.num_key_value_heads;
    const int64_t rows = params.q->size(0);
    const float scale = 1.0f / std::sqrt(static_cast<float>(params.head_dim));
    const size_t kv_token_bytes = static_cast<size_t>(kv_size) * sizeof(float);

    for (int64_t row = 0; row < rows; ++row)
    {
        const int32_t seq_index = seq_index_ptr[row];
        const int32_t position = positions_ptr[row];
        const int32_t block_id =
            block_id_for(params, block_ptr, num_seqs, max_blocks_per_seq, seq_index, position);
        const int32_t token_offset = position % metadata.block_size_tokens;
        const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(kv_size);
        float* key_dst = k_block_ptr(kv_cache, block_id)
            + static_cast<size_t>(token_offset) * static_cast<size_t>(kv_size);
        float* value_dst = v_block_ptr(kv_cache, block_id, metadata.block_size_tokens, kv_size)
            + static_cast<size_t>(token_offset) * static_cast<size_t>(kv_size);
        std::memcpy(key_dst, k_ptr + row_offset, kv_token_bytes);
        std::memcpy(value_dst, v_ptr + row_offset, kv_token_bytes);
    }

    std::vector<float> scores;
    for (int64_t row = 0; row < rows; ++row)
    {
        const int32_t seq_index = seq_index_ptr[row];
        const int32_t target_position = positions_ptr[row];
        if (target_position < 0 || target_position >= context_ptr[seq_index])
        {
            throw std::runtime_error("llama_attention: target position exceeds context length.");
        }

        const int32_t context_len = target_position + 1;
        scores.assign(static_cast<size_t>(context_len), -std::numeric_limits<float>::infinity());
        const size_t q_row_offset = static_cast<size_t>(row) * static_cast<size_t>(q_hidden_size);
        for (int32_t q_head = 0; q_head < params.num_attention_heads; ++q_head)
        {
            const int32_t kv_head = q_head / group_size;
            float max_score = -std::numeric_limits<float>::infinity();
            for (int32_t src_pos = 0; src_pos <= target_position; ++src_pos)
            {
                const int32_t block_id =
                    block_id_for(params, block_ptr, num_seqs, max_blocks_per_seq, seq_index, src_pos);
                const int32_t token_offset = src_pos % metadata.block_size_tokens;
                const float* key_base = k_block_ptr(kv_cache, block_id)
                    + static_cast<size_t>(token_offset) * static_cast<size_t>(kv_size);

                float score = 0.0f;
                for (int32_t dim = 0; dim < params.head_dim; ++dim)
                {
                    const size_t q_index = q_row_offset + static_cast<size_t>(q_head * params.head_dim + dim);
                    const size_t k_index = static_cast<size_t>(kv_head * params.head_dim + dim);
                    score += q_ptr[q_index] * key_base[k_index];
                }
                score *= scale;
                scores[static_cast<size_t>(src_pos)] = score;
                max_score = std::max(max_score, score);
            }

            float score_sum = 0.0f;
            for (int32_t src_pos = 0; src_pos <= target_position; ++src_pos)
            {
                const float exp_score = std::exp(scores[static_cast<size_t>(src_pos)] - max_score);
                scores[static_cast<size_t>(src_pos)] = exp_score;
                score_sum += exp_score;
            }
            if (score_sum <= 0.0f)
            {
                throw std::runtime_error("llama_attention: no paged causal source tokens.");
            }

            for (int32_t dim = 0; dim < params.head_dim; ++dim)
            {
                float value = 0.0f;
                for (int32_t src_pos = 0; src_pos <= target_position; ++src_pos)
                {
                    const int32_t block_id =
                        block_id_for(params, block_ptr, num_seqs, max_blocks_per_seq, seq_index, src_pos);
                    const int32_t token_offset = src_pos % metadata.block_size_tokens;
                    const float* value_base =
                        v_block_ptr(kv_cache, block_id, metadata.block_size_tokens, kv_size)
                        + static_cast<size_t>(token_offset) * static_cast<size_t>(kv_size);
                    const size_t v_index = static_cast<size_t>(kv_head * params.head_dim + dim);
                    value += (scores[static_cast<size_t>(src_pos)] / score_sum) * value_base[v_index];
                }
                const size_t out_index = q_row_offset + static_cast<size_t>(q_head * params.head_dim + dim);
                out_ptr[out_index] = value;
            }
        }
    }
}

} // namespace ops
} // namespace tiny_llm
