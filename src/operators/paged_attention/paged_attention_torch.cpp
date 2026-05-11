#include "paged_attention_internal.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/runtime/kv_cache.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace tiny_llm {
namespace ops {

namespace {

Tensor kv_block_tensor(KVCache& kv_cache,
                       int32_t block_id,
                       int32_t block_size_tokens,
                       int32_t kv_size,
                       bool value_block)
{
    void* block = kv_cache.block_ptr(block_id);
    if (block == nullptr)
    {
        throw std::runtime_error("llama_attention: KV block pointer is null.");
    }

    float* block_ptr = static_cast<float*>(block);
    if (value_block)
    {
        block_ptr += static_cast<size_t>(block_size_tokens) * static_cast<size_t>(kv_size);
    }

    return torch::from_blob(
        block_ptr,
        {block_size_tokens, kv_size},
        torch::TensorOptions().dtype(torch::kFloat32).device(kv_cache.device()));
}

void run_direct_attention_torch(const LlamaAttentionParams& params)
{
    Tensor positions_cpu;
    const int32_t* positions_ptr = cpu_int_ptr(*params.positions, positions_cpu);
    const int32_t group_size = params.num_attention_heads / params.num_key_value_heads;
    const int64_t rows = params.q->size(0);
    const float scale = 1.0f / std::sqrt(static_cast<float>(params.head_dim));

    for (int64_t row = 0; row < rows; ++row)
    {
        const int32_t target_position = positions_ptr[row];
        for (int32_t q_head = 0; q_head < params.num_attention_heads; ++q_head)
        {
            const int32_t kv_head = q_head / group_size;
            std::vector<int64_t> source_rows;
            source_rows.reserve(static_cast<size_t>(row + 1));
            for (int64_t src = 0; src < rows; ++src)
            {
                if (src <= row && positions_ptr[src] <= target_position)
                {
                    source_rows.push_back(src);
                }
            }
            if (source_rows.empty())
            {
                throw std::runtime_error("llama_attention: no causal source tokens.");
            }

            Tensor source_index = torch::tensor(
                source_rows,
                torch::TensorOptions().dtype(torch::kInt64).device(params.q->device()));
            Tensor k_context = params.k->index_select(0, source_index)
                .narrow(1, kv_head * params.head_dim, params.head_dim);
            Tensor v_context = params.v->index_select(0, source_index)
                .narrow(1, kv_head * params.head_dim, params.head_dim);
            Tensor q_head_tensor = params.q->narrow(0, row, 1)
                .narrow(1, q_head * params.head_dim, params.head_dim);
            Tensor probs = torch::softmax(torch::matmul(q_head_tensor, k_context.transpose(0, 1)) * scale, -1);
            Tensor value = torch::matmul(probs, v_context);
            params.out->narrow(0, row, 1)
                .narrow(1, q_head * params.head_dim, params.head_dim)
                .copy_(value);
        }
    }
}

void run_paged_attention_torch_reference(const LlamaAttentionParams& params)
{
    KVCache& kv_cache = *params.ctx->kv();
    const PagedAttentionRuntimeMetadata& metadata = *params.metadata;
    const Tensor& seq_indices = *metadata.seq_indices;
    const Tensor& context_lens = *metadata.context_lens;
    const Tensor& block_tables = *metadata.block_tables;

    Tensor positions_cpu;
    Tensor seq_indices_cpu;
    Tensor context_lens_cpu;
    Tensor block_tables_cpu;
    const int32_t* positions_ptr = cpu_int_ptr(*params.positions, positions_cpu);
    const int32_t* seq_index_ptr = cpu_int_ptr(seq_indices, seq_indices_cpu);
    const int32_t* context_ptr = cpu_int_ptr(context_lens, context_lens_cpu);
    const int32_t* block_ptr = cpu_int_ptr(block_tables, block_tables_cpu);

    const std::vector<int64_t> block_shape = tensor_shape(block_tables);
    const int64_t num_seqs = block_shape[1];
    const int64_t max_blocks_per_seq = block_shape[2];
    const int64_t rows = params.q->size(0);
    const int32_t kv_size = kv_hidden_size(params.num_key_value_heads, params.head_dim);
    const int32_t group_size = params.num_attention_heads / params.num_key_value_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(params.head_dim));

    auto block_id_for = [&](int32_t seq_index, int32_t position) -> int32_t {
        if (seq_index < 0 || seq_index >= num_seqs)
        {
            throw std::runtime_error("llama_attention: seq index out of range.");
        }
        const int32_t logical_block = position / metadata.block_size_tokens;
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
    };

    for (int64_t row = 0; row < rows; ++row)
    {
        const int32_t seq_index = seq_index_ptr[row];
        const int32_t position = positions_ptr[row];
        const int32_t block_id = block_id_for(seq_index, position);
        const int32_t token_offset = position % metadata.block_size_tokens;
        kv_block_tensor(kv_cache, block_id, metadata.block_size_tokens, kv_size, false)
            .narrow(0, token_offset, 1)
            .copy_(params.k->narrow(0, row, 1));
        kv_block_tensor(kv_cache, block_id, metadata.block_size_tokens, kv_size, true)
            .narrow(0, token_offset, 1)
            .copy_(params.v->narrow(0, row, 1));
    }

    for (int64_t row = 0; row < rows; ++row)
    {
        const int32_t seq_index = seq_index_ptr[row];
        const int32_t target_position = positions_ptr[row];
        if (target_position < 0 || target_position >= context_ptr[seq_index])
        {
            throw std::runtime_error("llama_attention: target position exceeds context length.");
        }

        const int32_t context_len = target_position + 1;
        Tensor k_context = torch::empty(
            {context_len, kv_size},
            torch::TensorOptions().dtype(torch::kFloat32).device(params.q->device()));
        Tensor v_context = torch::empty_like(k_context);
        for (int32_t src_pos = 0; src_pos <= target_position; ++src_pos)
        {
            const int32_t block_id = block_id_for(seq_index, src_pos);
            const int32_t token_offset = src_pos % metadata.block_size_tokens;
            k_context.narrow(0, src_pos, 1).copy_(
                kv_block_tensor(kv_cache, block_id, metadata.block_size_tokens, kv_size, false)
                    .narrow(0, token_offset, 1));
            v_context.narrow(0, src_pos, 1).copy_(
                kv_block_tensor(kv_cache, block_id, metadata.block_size_tokens, kv_size, true)
                    .narrow(0, token_offset, 1));
        }

        for (int32_t q_head = 0; q_head < params.num_attention_heads; ++q_head)
        {
            const int32_t kv_head = q_head / group_size;
            Tensor k_head_context = k_context.narrow(1, kv_head * params.head_dim, params.head_dim);
            Tensor v_head_context = v_context.narrow(1, kv_head * params.head_dim, params.head_dim);
            Tensor q_head_tensor = params.q->narrow(0, row, 1)
                .narrow(1, q_head * params.head_dim, params.head_dim);
            Tensor probs = torch::softmax(torch::matmul(q_head_tensor, k_head_context.transpose(0, 1)) * scale, -1);
            Tensor value = torch::matmul(probs, v_head_context);
            params.out->narrow(0, row, 1)
                .narrow(1, q_head * params.head_dim, params.head_dim)
                .copy_(value);
        }
    }
}

} // namespace

void run_torch_reference_attention(const LlamaAttentionParams& params)
{
    if (has_paged_kv_cache(params))
    {
        run_paged_attention_torch_reference(params);
        return;
    }
    run_direct_attention_torch(params);
}

bool try_run_cuda_optimized_attention(const LlamaAttentionParams& params)
{
#if TINYLLM_ENABLE_CUDA
    if (!has_paged_kv_cache(params))
    {
        return false;
    }
    if (!params.q->device().is_cuda() || !params.k->device().is_cuda()
        || !params.v->device().is_cuda() || !params.out->device().is_cuda())
    {
        return false;
    }

    if (!params.positions->device().is_cuda()
        || !params.metadata->seq_indices->device().is_cuda()
        || !params.metadata->context_lens->device().is_cuda()
        || !params.metadata->block_tables->device().is_cuda())
    {
        return false;
    }

    KVCache& kv_cache = *params.ctx->kv();
    void* base = kv_cache.block_pool_base();
    if (base == nullptr || kv_cache.total_block_count() == 0)
    {
        return false;
    }

    const Tensor& block_tables = *params.metadata->block_tables;
    const std::vector<int64_t> block_shape = tensor_shape(block_tables);
    cuda::launch_paged_attention_f32(
        static_cast<const float*>(tensor_data(*params.q)),
        static_cast<const float*>(tensor_data(*params.k)),
        static_cast<const float*>(tensor_data(*params.v)),
        static_cast<float*>(tensor_data(*params.out)),
        params.positions->data_ptr<int32_t>(),
        params.metadata->seq_indices->data_ptr<int32_t>(),
        params.metadata->context_lens->data_ptr<int32_t>(),
        params.metadata->block_tables->data_ptr<int32_t>(),
        static_cast<float*>(base),
        params.q->size(0),
        block_shape[1],
        block_shape[2],
        static_cast<int64_t>(kv_cache.total_block_count()),
        static_cast<int64_t>(kv_cache.block_size_bytes()),
        params.metadata->block_size_tokens,
        params.layer_id,
        params.num_attention_heads,
        params.num_key_value_heads,
        params.head_dim,
        params.ctx->stream());
    return true;
#else
    (void)params;
    return false;
#endif
}

} // namespace ops
} // namespace tiny_llm
