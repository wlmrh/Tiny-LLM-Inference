#include "paged_attention_internal.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/runtime/kv_cache.h"

#include <ATen/ops/cat.h>
#include <ATen/ops/scaled_dot_product_attention.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#if TINYLLM_ENABLE_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif

namespace tiny_llm
{
namespace ops
{

namespace
{

Tensor kv_block_tensor(KVCache &kv_cache, int32_t block_id, int32_t block_size_tokens, int32_t kv_size,
                       bool value_block)
{
    void *block = kv_cache.block_ptr(block_id);
    if (block == nullptr)
    {
        throw std::runtime_error("llama_attention: KV block pointer is null.");
    }

    auto *block_ptr = static_cast<uint8_t *>(block);
    const size_t element_bytes = runtime_dtype_size(kv_cache.dtype());
    if (value_block)
    {
        block_ptr += static_cast<size_t>(block_size_tokens) * static_cast<size_t>(kv_size) * element_bytes;
    }

    const c10::ScalarType scalar_type =
        kv_cache.dtype() == RuntimeDType::kBFloat16 ? torch::kBFloat16 : torch::kFloat32;
    return torch::from_blob(block_ptr, {block_size_tokens, kv_size},
                            torch::TensorOptions().dtype(scalar_type).device(kv_cache.device()));
}

#if TINYLLM_ENABLE_CUDA
constexpr int64_t kSdpaMinQueryTokens = 64;
// Keep batched/math SDPA intermediates within the per-step memory budget.
constexpr int64_t kMaxBatchedSdpaScoreElements = 16 * 1024 * 1024;
constexpr int64_t kSdpaQueryTileTokens = 512;

bool collect_query_segments(const LlamaAttentionParams &params, int64_t rows, int64_t num_seqs,
                            std::vector<PagedAttentionQuerySegment> &segments)
{
    if (params.metadata == nullptr || !params.metadata->query_segments_valid ||
        params.metadata->query_segments == nullptr || params.metadata->query_segment_count <= 0)
    {
        return false;
    }

    segments.clear();
    std::vector<bool> seen(static_cast<size_t>(num_seqs), false);
    int64_t next_row = 0;
    for (int64_t index = 0; index < params.metadata->query_segment_count; ++index)
    {
        const PagedAttentionQuerySegment &segment = params.metadata->query_segments[index];
        if (segment.row_start != next_row || segment.seq_index < 0 || segment.seq_index >= num_seqs ||
            segment.query_start_position < 0 || segment.query_length <= 0 ||
            next_row + static_cast<int64_t>(segment.query_length) > rows ||
            seen[static_cast<size_t>(segment.seq_index)])
        {
            segments.clear();
            return false;
        }
        seen[static_cast<size_t>(segment.seq_index)] = true;
        segments.push_back(segment);
        next_row += segment.query_length;
    }
    if (next_row != rows)
    {
        segments.clear();
        return false;
    }
    return !segments.empty();
}

bool has_bulk_query(const std::vector<PagedAttentionQuerySegment> &segments)
{
    for (const PagedAttentionQuerySegment &segment : segments)
    {
        if (segment.query_length >= kSdpaMinQueryTokens)
        {
            return true;
        }
    }
    return false;
}

void ensure_contiguous_kv_scratch(PagedAttentionRuntimeScratch &scratch, const Tensor &like, int64_t tokens,
                                  int64_t kv_size)
{
    const bool needs_allocation = !scratch.contiguous_k.defined() || !scratch.contiguous_v.defined() ||
                                  scratch.contiguous_k.dim() != 2 || scratch.contiguous_v.dim() != 2 ||
                                  scratch.contiguous_k.size(0) < tokens || scratch.contiguous_v.size(0) < tokens ||
                                  scratch.contiguous_k.size(1) != kv_size || scratch.contiguous_v.size(1) != kv_size ||
                                  scratch.contiguous_k.device() != like.device() ||
                                  scratch.contiguous_v.device() != like.device() ||
                                  scratch.contiguous_k.scalar_type() != like.scalar_type() ||
                                  scratch.contiguous_v.scalar_type() != like.scalar_type();
    if (needs_allocation)
    {
        scratch.contiguous_k = torch::empty({tokens, kv_size}, like.options());
        scratch.contiguous_v = torch::empty({tokens, kv_size}, like.options());
    }
}

void gather_paged_kv_context(const LlamaAttentionParams &params, KVCache &kv_cache,
                             const PagedAttentionQuerySegment &segment, int64_t num_seqs,
                             int64_t max_blocks_per_seq, PagedAttentionRuntimeScratch &scratch, Tensor &k_context,
                             Tensor &v_context)
{
    const PagedAttentionRuntimeMetadata &metadata = *params.metadata;
    const int64_t context_len =
        static_cast<int64_t>(segment.query_start_position) + static_cast<int64_t>(segment.query_length);
    const int64_t required_blocks =
        (context_len + static_cast<int64_t>(metadata.block_size_tokens) - 1) / metadata.block_size_tokens;
    const int32_t kv_size = kv_hidden_size(params.num_key_value_heads, params.head_dim);
    const int64_t padded_tokens = required_blocks * static_cast<int64_t>(metadata.block_size_tokens);

    ensure_contiguous_kv_scratch(scratch, *params.k, padded_tokens, kv_size);
    std::vector<Tensor> key_blocks;
    std::vector<Tensor> value_blocks;
    key_blocks.reserve(static_cast<size_t>(required_blocks));
    value_blocks.reserve(static_cast<size_t>(required_blocks));
    for (int64_t logical_block = 0; logical_block < required_blocks; ++logical_block)
    {
        const int64_t table_index = static_cast<int64_t>(params.layer_id) * num_seqs * max_blocks_per_seq +
                                    static_cast<int64_t>(segment.seq_index) * max_blocks_per_seq + logical_block;
        if (table_index < 0 || table_index >= metadata.host_block_table_count)
        {
            throw std::runtime_error("llama_attention: host block table does not cover the SDPA context.");
        }
        const int32_t block_id = metadata.host_block_tables[table_index];
        if (block_id < 0 || block_id >= static_cast<int32_t>(kv_cache.total_block_count()))
        {
            throw std::runtime_error("llama_attention: invalid physical block id in host block table.");
        }
        key_blocks.push_back(
            kv_block_tensor(kv_cache, block_id, metadata.block_size_tokens, kv_size, /*value_block=*/false));
        value_blocks.push_back(
            kv_block_tensor(kv_cache, block_id, metadata.block_size_tokens, kv_size, /*value_block=*/true));
    }

    Tensor key_out = scratch.contiguous_k.narrow(0, 0, padded_tokens);
    Tensor value_out = scratch.contiguous_v.narrow(0, 0, padded_tokens);
    at::cat_out(key_out, key_blocks, 0);
    at::cat_out(value_out, value_blocks, 0);
    k_context = key_out.narrow(0, 0, context_len);
    v_context = value_out.narrow(0, 0, context_len);
}

Tensor query_as_sdpa(const Tensor &query, int64_t row_start, int64_t query_length, int32_t num_heads,
                     int32_t head_dim)
{
    return query.narrow(0, row_start, query_length)
        .view({query_length, num_heads, head_dim})
        .permute({1, 0, 2})
        .unsqueeze(0)
        .contiguous();
}

Tensor kv_as_sdpa(const Tensor &kv, int64_t tokens, int32_t num_heads, int32_t head_dim)
{
    return kv.view({tokens, num_heads, head_dim}).permute({1, 0, 2}).unsqueeze(0).contiguous();
}

void copy_sdpa_output(const Tensor &attended, Tensor &out, int64_t row_start, int64_t query_length,
                      int64_t attention_hidden)
{
    out.narrow(0, row_start, query_length)
        .copy_(attended.squeeze(0).permute({1, 0, 2}).contiguous().view({query_length, attention_hidden}));
}

void run_sdpa_prefill_segment(const LlamaAttentionParams &params, KVCache &kv_cache,
                              const PagedAttentionQuerySegment &segment, int64_t segment_index, int64_t num_seqs,
                              int64_t max_blocks_per_seq, PagedAttentionRuntimeScratch &scratch)
{
    const int64_t query_length = segment.query_length;
    const int64_t context_len =
        static_cast<int64_t>(segment.query_start_position) + static_cast<int64_t>(segment.query_length);
    const int64_t attention_hidden = attention_hidden_size(params.num_attention_heads, params.head_dim);

    Tensor key_context;
    Tensor value_context;
    if (segment.query_start_position == 0)
    {
        key_context = params.k->narrow(0, segment.row_start, query_length);
        value_context = params.v->narrow(0, segment.row_start, query_length);
    }
    else
    {
        gather_paged_kv_context(params, kv_cache, segment, num_seqs, max_blocks_per_seq, scratch, key_context,
                                value_context);
    }
    Tensor key = kv_as_sdpa(key_context, context_len, params.num_key_value_heads, params.head_dim);
    Tensor value = kv_as_sdpa(value_context, context_len, params.num_key_value_heads, params.head_dim);

    const bool needs_offset_mask = segment.query_start_position > 0 || query_length > kSdpaQueryTileTokens;
    Tensor mask;
    if (needs_offset_mask)
    {
        if (scratch.offset_causal_masks.size() < static_cast<size_t>(params.metadata->query_segment_count))
        {
            scratch.offset_causal_masks.resize(static_cast<size_t>(params.metadata->query_segment_count));
        }
        Tensor &cached_mask = scratch.offset_causal_masks[static_cast<size_t>(segment_index)];
        if (!cached_mask.defined())
        {
            const auto index_options = torch::TensorOptions().dtype(torch::kInt64).device(params.q->device());
            Tensor key_positions = torch::arange(context_len, index_options).unsqueeze(0);
            Tensor query_positions =
                torch::arange(static_cast<int64_t>(segment.query_start_position), context_len, index_options)
                    .unsqueeze(1);
            cached_mask = query_positions.ge(key_positions);
        }
        mask = cached_mask;
    }

    for (int64_t tile_start = 0; tile_start < query_length; tile_start += kSdpaQueryTileTokens)
    {
        const int64_t tile_length = std::min(kSdpaQueryTileTokens, query_length - tile_start);
        Tensor query = query_as_sdpa(*params.q, segment.row_start + tile_start, tile_length,
                                     params.num_attention_heads, params.head_dim);
        std::optional<Tensor> tile_mask = std::nullopt;
        if (mask.defined())
        {
            tile_mask = mask.narrow(0, tile_start, tile_length);
        }
        Tensor attended =
            at::scaled_dot_product_attention(query, key, value, tile_mask, 0.0, !mask.defined(), std::nullopt,
                                             params.num_attention_heads != params.num_key_value_heads);
        copy_sdpa_output(attended, *params.out, segment.row_start + tile_start, tile_length, attention_hidden);
    }
}

void run_paged_decode_segment(const LlamaAttentionParams &params, const PagedAttentionQuerySegment &segment,
                              const float *kv_pool_base, int64_t num_seqs, int64_t max_blocks_per_seq,
                              int64_t num_blocks, int64_t block_size_bytes)
{
    const int64_t attention_hidden = attention_hidden_size(params.num_attention_heads, params.head_dim);
    const auto *query = static_cast<const float *>(tensor_data(*params.q)) +
                        segment.row_start * static_cast<int64_t>(attention_hidden);
    auto *out = static_cast<float *>(tensor_data(*params.out)) +
                segment.row_start * static_cast<int64_t>(attention_hidden);
    cuda::launch_paged_attention_query_f32(
        query, out, params.positions->data_ptr<int32_t>() + segment.row_start,
        params.metadata->seq_indices->data_ptr<int32_t>() + segment.row_start,
        params.metadata->context_lens->data_ptr<int32_t>(), params.metadata->block_tables->data_ptr<int32_t>(),
        kv_pool_base, segment.query_length, num_seqs, max_blocks_per_seq, num_blocks, block_size_bytes,
        params.metadata->block_size_tokens, params.layer_id, params.num_attention_heads, params.num_key_value_heads,
        params.head_dim, params.ctx->stream());
}

bool try_run_segmented_sdpa_cuda(const LlamaAttentionParams &params, KVCache &kv_cache, int64_t rows,
                                 int64_t num_seqs, int64_t max_blocks_per_seq, int64_t num_blocks,
                                 int64_t block_size_bytes)
{
    std::vector<PagedAttentionQuerySegment> segments;
    if (!collect_query_segments(params, rows, num_seqs, segments) || !has_bulk_query(segments))
    {
        return false;
    }

    bool needs_host_block_table = false;
    for (const PagedAttentionQuerySegment &segment : segments)
    {
        needs_host_block_table =
            needs_host_block_table ||
            (segment.query_length >= kSdpaMinQueryTokens && segment.query_start_position > 0);
    }
    if (needs_host_block_table &&
        (params.metadata->host_block_tables == nullptr ||
         params.metadata->host_block_table_count != params.metadata->block_tables->numel()))
    {
        return false;
    }

    std::optional<c10::cuda::CUDAStreamGuard> stream_guard;
    if (params.ctx->stream() != nullptr)
    {
        stream_guard.emplace(c10::cuda::getStreamFromExternal(
            params.ctx->stream(), static_cast<c10::DeviceIndex>(params.q->device().index())));
    }

    const int32_t kv_size = kv_hidden_size(params.num_key_value_heads, params.head_dim);
    cuda::launch_write_paged_kv_cache_f32(
        static_cast<const float *>(tensor_data(*params.k)), static_cast<const float *>(tensor_data(*params.v)),
        params.positions->data_ptr<int32_t>(), params.metadata->seq_indices->data_ptr<int32_t>(),
        params.metadata->block_tables->data_ptr<int32_t>(), static_cast<float *>(kv_cache.block_pool_base()), rows,
        num_seqs, max_blocks_per_seq, num_blocks, block_size_bytes, params.metadata->block_size_tokens, params.layer_id,
        kv_size, params.ctx->stream());

    torch::NoGradGuard no_grad;
    PagedAttentionRuntimeScratch local_scratch;
    PagedAttentionRuntimeScratch &scratch =
        params.metadata->scratch != nullptr ? *params.metadata->scratch : local_scratch;

    const int64_t segment_count = static_cast<int64_t>(segments.size());
    bool can_run_batched_prefill = segment_count > 1;
    const int64_t first_len = can_run_batched_prefill ? segments.front().query_length : 0;
    const int64_t estimated_score_elements =
        segment_count * static_cast<int64_t>(params.num_attention_heads) * first_len * first_len;
    can_run_batched_prefill =
        can_run_batched_prefill && estimated_score_elements <= kMaxBatchedSdpaScoreElements;
    for (int64_t index = 0; can_run_batched_prefill && index < segment_count; ++index)
    {
        const PagedAttentionQuerySegment &segment = segments[static_cast<size_t>(index)];
        can_run_batched_prefill = segment.query_start_position == 0 &&
                                  segment.query_length >= kSdpaMinQueryTokens &&
                                  static_cast<int64_t>(segment.query_length) == first_len &&
                                  segment.row_start == index * first_len;
    }
    if (can_run_batched_prefill)
    {
        const int64_t attention_hidden = attention_hidden_size(params.num_attention_heads, params.head_dim);
        Tensor query = params.q->view({segment_count, first_len, params.num_attention_heads, params.head_dim})
                           .permute({0, 2, 1, 3})
                           .contiguous();
        Tensor key = params.k->view({segment_count, first_len, params.num_key_value_heads, params.head_dim})
                         .permute({0, 2, 1, 3})
                         .contiguous();
        Tensor value = params.v->view({segment_count, first_len, params.num_key_value_heads, params.head_dim})
                           .permute({0, 2, 1, 3})
                           .contiguous();
        Tensor attended =
            at::scaled_dot_product_attention(query, key, value, std::nullopt, 0.0, true, std::nullopt,
                                             params.num_attention_heads != params.num_key_value_heads);
        params.out->copy_(attended.permute({0, 2, 1, 3}).contiguous().view({rows, attention_hidden}));
        return true;
    }

    for (int64_t index = 0; index < segment_count; ++index)
    {
        const PagedAttentionQuerySegment &segment = segments[static_cast<size_t>(index)];
        if (segment.query_length >= kSdpaMinQueryTokens)
        {
            run_sdpa_prefill_segment(params, kv_cache, segment, index, num_seqs, max_blocks_per_seq, scratch);
        }
        else
        {
            run_paged_decode_segment(params, segment, static_cast<const float *>(kv_cache.block_pool_base()), num_seqs,
                                     max_blocks_per_seq, num_blocks, block_size_bytes);
        }
    }
    return true;
}
#endif

void run_direct_attention_torch(const LlamaAttentionParams &params)
{
    Tensor positions_cpu;
    const int32_t *positions_ptr = cpu_int_ptr(*params.positions, positions_cpu);
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

            Tensor source_index =
                torch::tensor(source_rows, torch::TensorOptions().dtype(torch::kInt64).device(params.q->device()));
            Tensor k_context =
                params.k->index_select(0, source_index).narrow(1, kv_head * params.head_dim, params.head_dim);
            Tensor v_context =
                params.v->index_select(0, source_index).narrow(1, kv_head * params.head_dim, params.head_dim);
            Tensor q_head_tensor = params.q->narrow(0, row, 1).narrow(1, q_head * params.head_dim, params.head_dim);
            Tensor probs = torch::softmax(torch::matmul(q_head_tensor, k_context.transpose(0, 1)) * scale, -1);
            Tensor value = torch::matmul(probs, v_context);
            params.out->narrow(0, row, 1).narrow(1, q_head * params.head_dim, params.head_dim).copy_(value);
        }
    }
}

void run_paged_attention_torch_reference(const LlamaAttentionParams &params)
{
    KVCache &kv_cache = *params.ctx->kv();
    const PagedAttentionRuntimeMetadata &metadata = *params.metadata;
    const Tensor &seq_indices = *metadata.seq_indices;
    const Tensor &context_lens = *metadata.context_lens;
    const Tensor &block_tables = *metadata.block_tables;

    Tensor positions_cpu;
    Tensor seq_indices_cpu;
    Tensor context_lens_cpu;
    Tensor block_tables_cpu;
    const int32_t *positions_ptr = cpu_int_ptr(*params.positions, positions_cpu);
    const int32_t *seq_index_ptr = cpu_int_ptr(seq_indices, seq_indices_cpu);
    const int32_t *context_ptr = cpu_int_ptr(context_lens, context_lens_cpu);
    const int32_t *block_ptr = cpu_int_ptr(block_tables, block_tables_cpu);

    const std::vector<int64_t> block_shape = tensor_shape(block_tables);
    const int64_t num_seqs = block_shape[1];
    const int64_t max_blocks_per_seq = block_shape[2];
    const int64_t rows = params.q->size(0);
    const int32_t kv_size = kv_hidden_size(params.num_key_value_heads, params.head_dim);
    const int32_t group_size = params.num_attention_heads / params.num_key_value_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(params.head_dim));

    auto block_id_for = [&](int32_t seq_index, int32_t position) -> int32_t
    {
        if (seq_index < 0 || seq_index >= num_seqs)
        {
            throw std::runtime_error("llama_attention: seq index out of range.");
        }
        const int32_t logical_block = position / metadata.block_size_tokens;
        if (logical_block < 0 || logical_block >= max_blocks_per_seq)
        {
            throw std::runtime_error("llama_attention: logical block out of range.");
        }
        const int64_t index = static_cast<int64_t>(params.layer_id) * num_seqs * max_blocks_per_seq +
                              static_cast<int64_t>(seq_index) * max_blocks_per_seq + logical_block;
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
        Tensor k_context = torch::empty({context_len, kv_size},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(params.q->device()));
        Tensor v_context = torch::empty_like(k_context);
        for (int32_t src_pos = 0; src_pos <= target_position; ++src_pos)
        {
            const int32_t block_id = block_id_for(seq_index, src_pos);
            const int32_t token_offset = src_pos % metadata.block_size_tokens;
            k_context.narrow(0, src_pos, 1)
                .copy_(kv_block_tensor(kv_cache, block_id, metadata.block_size_tokens, kv_size, false)
                           .narrow(0, token_offset, 1));
            v_context.narrow(0, src_pos, 1)
                .copy_(kv_block_tensor(kv_cache, block_id, metadata.block_size_tokens, kv_size, true)
                           .narrow(0, token_offset, 1));
        }

        for (int32_t q_head = 0; q_head < params.num_attention_heads; ++q_head)
        {
            const int32_t kv_head = q_head / group_size;
            Tensor k_head_context = k_context.narrow(1, kv_head * params.head_dim, params.head_dim);
            Tensor v_head_context = v_context.narrow(1, kv_head * params.head_dim, params.head_dim);
            Tensor q_head_tensor = params.q->narrow(0, row, 1).narrow(1, q_head * params.head_dim, params.head_dim);
            Tensor probs = torch::softmax(torch::matmul(q_head_tensor, k_head_context.transpose(0, 1)) * scale, -1);
            Tensor value = torch::matmul(probs, v_head_context);
            params.out->narrow(0, row, 1).narrow(1, q_head * params.head_dim, params.head_dim).copy_(value);
        }
    }
}

} // namespace

void run_torch_reference_attention(const LlamaAttentionParams &params)
{
    if (has_paged_kv_cache(params))
    {
        run_paged_attention_torch_reference(params);
        return;
    }
    run_direct_attention_torch(params);
}

bool try_run_cuda_optimized_attention(const LlamaAttentionParams &params)
{
#if TINYLLM_ENABLE_CUDA
    if (!has_paged_kv_cache(params))
    {
        return false;
    }
    if (!params.q->device().is_cuda() || !params.k->device().is_cuda() || !params.v->device().is_cuda() ||
        !params.out->device().is_cuda())
    {
        return false;
    }

    if (!params.positions->device().is_cuda() || !params.metadata->seq_indices->device().is_cuda() ||
        !params.metadata->context_lens->device().is_cuda() || !params.metadata->block_tables->device().is_cuda())
    {
        return false;
    }

    KVCache &kv_cache = *params.ctx->kv();
    if (kv_cache.dtype() != RuntimeDType::kFloat32)
    {
        const Tensor &block_tables = *params.metadata->block_tables;
        const std::vector<int64_t> block_shape = tensor_shape(block_tables);
        cuda::launch_paged_attention_bf16_kv(
            static_cast<const float *>(tensor_data(*params.q)), static_cast<const float *>(tensor_data(*params.k)),
            static_cast<const float *>(tensor_data(*params.v)), static_cast<float *>(tensor_data(*params.out)),
            params.positions->data_ptr<int32_t>(), params.metadata->seq_indices->data_ptr<int32_t>(),
            params.metadata->context_lens->data_ptr<int32_t>(), params.metadata->block_tables->data_ptr<int32_t>(),
            kv_cache.block_pool_base(), params.q->size(0), block_shape[1], block_shape[2],
            static_cast<int64_t>(kv_cache.total_block_count()), static_cast<int64_t>(kv_cache.block_size_bytes()),
            params.metadata->block_size_tokens, params.layer_id, params.num_attention_heads, params.num_key_value_heads,
            params.head_dim, params.ctx->stream());
        return true;
    }
    void *base = kv_cache.block_pool_base();
    if (base == nullptr || kv_cache.total_block_count() == 0)
    {
        return false;
    }

    const Tensor &block_tables = *params.metadata->block_tables;
    const std::vector<int64_t> block_shape = tensor_shape(block_tables);
    const int64_t rows = params.q->size(0);
    if (try_run_segmented_sdpa_cuda(params, kv_cache, rows, block_shape[1], block_shape[2],
                                    static_cast<int64_t>(kv_cache.total_block_count()),
                                    static_cast<int64_t>(kv_cache.block_size_bytes())))
    {
        return true;
    }

    cuda::launch_paged_attention_f32(
        static_cast<const float *>(tensor_data(*params.q)), static_cast<const float *>(tensor_data(*params.k)),
        static_cast<const float *>(tensor_data(*params.v)), static_cast<float *>(tensor_data(*params.out)),
        params.positions->data_ptr<int32_t>(), params.metadata->seq_indices->data_ptr<int32_t>(),
        params.metadata->context_lens->data_ptr<int32_t>(), params.metadata->block_tables->data_ptr<int32_t>(),
        static_cast<float *>(base), rows, block_shape[1], block_shape[2],
        static_cast<int64_t>(kv_cache.total_block_count()), static_cast<int64_t>(kv_cache.block_size_bytes()),
        params.metadata->block_size_tokens, params.layer_id, params.num_attention_heads, params.num_key_value_heads,
        params.head_dim, params.ctx->stream());
    return true;
#else
    (void)params;
    return false;
#endif
}

} // namespace ops
} // namespace tiny_llm
