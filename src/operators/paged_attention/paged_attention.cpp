#include "tiny_llm/operators/paged_attention.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/execution_context.h"
#include "tiny_llm/runtime/kv_cache.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace tiny_llm {
namespace ops {

namespace {

thread_local PagedAttentionRuntimeMetadata g_runtime_metadata;

int32_t kv_hidden_size(int32_t num_key_value_heads, int32_t head_dim)
{
    return num_key_value_heads * head_dim;
}

int32_t hidden_size(int32_t num_attention_heads, int32_t head_dim)
{
    return num_attention_heads * head_dim;
}

void validate_cpu_tensor(const Tensor& tensor, const char* name)
{
    if (tensor.device().is_cuda())
    {
        throw std::runtime_error(std::string(name) + ": CUDA path is not implemented.");
    }
}

bool any_cuda(std::initializer_list<std::reference_wrapper<const Tensor>> tensors)
{
    for (const Tensor& tensor : tensors)
    {
        if (tensor.defined() && tensor.device().is_cuda())
        {
            return true;
        }
    }
    return false;
}

void validate_same_device(std::initializer_list<std::reference_wrapper<const Tensor>> tensors, const char* name)
{
    bool have_device = false;
    c10::Device device(c10::kCPU);
    for (const Tensor& tensor : tensors)
    {
        if (!tensor.defined())
        {
            throw std::runtime_error(std::string(name) + ": tensor must be defined.");
        }
        if (!have_device)
        {
            device = tensor.device();
            have_device = true;
            continue;
        }
        if (tensor.device() != device)
        {
            throw std::runtime_error(std::string(name) + ": tensors must be on the same device.");
        }
    }
}

Tensor tensor_to_cpu_contiguous(const Tensor& tensor)
{
    if (tensor.device().is_cpu())
    {
        return tensor.contiguous();
    }
    return tensor.to(c10::kCPU, /*non_blocking=*/false, /*copy=*/true).contiguous();
}

void validate_llama_attention_inputs(const Tensor& positions,
                                     const Tensor& q,
                                     const Tensor& k,
                                     const Tensor& v,
                                     const Tensor& out,
                                     int32_t num_attention_heads,
                                     int32_t num_key_value_heads,
                                     int32_t head_dim)
{
    if (num_attention_heads <= 0 || num_key_value_heads <= 0 || head_dim <= 0)
    {
        throw std::runtime_error("llama_attention: attention dimensions must be positive.");
    }
    if (num_attention_heads % num_key_value_heads != 0)
    {
        throw std::runtime_error("llama_attention: num_attention_heads must be divisible by num_key_value_heads.");
    }
    if (tensor_dtype(positions) != DType::kInt32
        || tensor_dtype(q) != DType::kFloat32
        || tensor_dtype(k) != DType::kFloat32
        || tensor_dtype(v) != DType::kFloat32
        || tensor_dtype(out) != DType::kFloat32)
    {
        throw std::runtime_error("llama_attention: dtype mismatch.");
    }
    if (q.dim() != 2)
    {
        throw std::runtime_error("llama_attention: q must be rank-2.");
    }

    const int64_t rows = q.size(0);
    if (positions.dim() != 1 || positions.size(0) != rows)
    {
        throw std::runtime_error("llama_attention: positions must be [rows].");
    }
    if (q.size(1) != hidden_size(num_attention_heads, head_dim))
    {
        throw std::runtime_error("llama_attention: q hidden size mismatch.");
    }
    if (k.dim() != 2 || v.dim() != 2 || out.dim() != 2
        || k.size(0) != rows || v.size(0) != rows || out.size(0) != rows
        || k.size(1) != kv_hidden_size(num_key_value_heads, head_dim)
        || v.size(1) != kv_hidden_size(num_key_value_heads, head_dim)
        || out.size(1) != hidden_size(num_attention_heads, head_dim))
    {
        throw std::runtime_error("llama_attention: k/v/out shape mismatch.");
    }
    if (tensor_data(positions) == nullptr
        || tensor_data(q) == nullptr
        || tensor_data(k) == nullptr
        || tensor_data(v) == nullptr
        || tensor_data(out) == nullptr)
    {
        throw std::runtime_error("llama_attention: data pointers must be non-null.");
    }
}

const int32_t* cpu_int_ptr(const Tensor& tensor, Tensor& storage)
{
    storage = tensor_to_cpu_contiguous(tensor);
    return storage.data_ptr<int32_t>();
}

void validate_runtime_metadata(const Tensor& q)
{
    if (!g_runtime_metadata.enabled)
    {
        return;
    }

    if (g_runtime_metadata.slot_mapping == nullptr
        || g_runtime_metadata.seq_indices == nullptr
        || g_runtime_metadata.context_lens == nullptr
        || g_runtime_metadata.block_tables == nullptr)
    {
        throw std::runtime_error("attention_paged: runtime metadata pointers must be non-null when enabled.");
    }

    const Tensor& slot_mapping = *g_runtime_metadata.slot_mapping;
    const Tensor& seq_indices = *g_runtime_metadata.seq_indices;
    const Tensor& context_lens = *g_runtime_metadata.context_lens;
    const Tensor& block_tables = *g_runtime_metadata.block_tables;

    if (tensor_dtype(slot_mapping) != DType::kInt32
        || tensor_dtype(seq_indices) != DType::kInt32
        || tensor_dtype(context_lens) != DType::kInt32
        || tensor_dtype(block_tables) != DType::kInt32)
    {
        throw std::runtime_error("attention_paged: runtime metadata tensors must be int32.");
    }

    const std::vector<int64_t> q_shape = tensor_shape(q);
    if (q_shape.empty())
    {
        throw std::runtime_error("attention_paged: q rank must be >= 1.");
    }
    const int64_t num_total_tokens = q_shape[0];

    const std::vector<int64_t> slot_shape = tensor_shape(slot_mapping);
    if (slot_shape.size() != 1 || slot_shape[0] != num_total_tokens)
    {
        throw std::runtime_error("attention_paged: slot_mapping must be rank-1 and align with q batch dimension.");
    }
    const std::vector<int64_t> seq_index_shape = tensor_shape(seq_indices);
    if (seq_index_shape.size() != 1 || seq_index_shape[0] != num_total_tokens)
    {
        throw std::runtime_error("attention_paged: seq_indices must be rank-1 and align with q batch dimension.");
    }

    const std::vector<int64_t> context_shape = tensor_shape(context_lens);
    if (context_shape.size() != 1)
    {
        throw std::runtime_error("attention_paged: context_lens must be rank-1 [num_seqs].");
    }

    const int64_t num_seqs = context_shape[0];
    if (num_seqs <= 0)
    {
        throw std::runtime_error("attention_paged: context_lens must contain at least one sequence.");
    }

    const std::vector<int64_t> block_shape = tensor_shape(block_tables);
    if (block_shape.size() != 3 || block_shape[1] != num_seqs)
    {
        throw std::runtime_error("attention_paged: block_tables must be rank-3 [num_layers, num_seqs, max_blocks_per_seq].");
    }
    if (block_shape[0] <= 0 || block_shape[2] <= 0)
    {
        throw std::runtime_error("attention_paged: block_tables dimensions must be positive.");
    }

    if (g_runtime_metadata.block_size_tokens <= 0)
    {
        throw std::runtime_error("attention_paged: block_size_tokens must be positive.");
    }

    Tensor slot_cpu;
    Tensor seq_indices_cpu;
    Tensor context_lens_cpu;
    Tensor block_tables_cpu;
    const int32_t* slot_ptr = cpu_int_ptr(slot_mapping, slot_cpu);
    const int32_t* seq_index_ptr = cpu_int_ptr(seq_indices, seq_indices_cpu);
    const int32_t* context_ptr = cpu_int_ptr(context_lens, context_lens_cpu);
    const int32_t* block_ptr = cpu_int_ptr(block_tables, block_tables_cpu);

    const int64_t num_layers = block_shape[0];
    const int64_t max_blocks_per_seq = block_shape[2];
    std::unordered_set<int32_t> known_block_ids;
    known_block_ids.reserve(static_cast<size_t>(num_layers * num_seqs * max_blocks_per_seq));

    for (int64_t seq = 0; seq < num_seqs; ++seq)
    {
        const int32_t context_len = context_ptr[seq];
        if (context_len < 0)
        {
            throw std::runtime_error("attention_paged: context_lens values must be non-negative.");
        }

        const int64_t required_blocks = context_len == 0
            ? 0
            : (static_cast<int64_t>(context_len) - 1)
                / static_cast<int64_t>(g_runtime_metadata.block_size_tokens)
                + 1;

        int64_t valid_blocks_in_row = 0;
        for (int64_t col = 0; col < max_blocks_per_seq; ++col)
        {
            const int32_t block_id = block_ptr[seq * max_blocks_per_seq + col];
            if (block_id < 0)
            {
                break;
            }
            known_block_ids.insert(block_id);
            ++valid_blocks_in_row;
        }

        if (valid_blocks_in_row < required_blocks)
        {
            throw std::runtime_error("attention_paged: block_tables row does not cover required context length.");
        }
    }

    for (int64_t token = 0; token < num_total_tokens; ++token)
    {
        const int32_t seq_index = seq_index_ptr[token];
        if (seq_index < 0 || seq_index >= num_seqs)
        {
            throw std::runtime_error("attention_paged: seq_indices values must reference valid sequences.");
        }
        const int32_t slot = slot_ptr[token];
        if (slot < 0)
        {
            throw std::runtime_error("attention_paged: slot_mapping values must be non-negative.");
        }

        const int32_t block_id = slot / g_runtime_metadata.block_size_tokens;
        if (known_block_ids.find(block_id) == known_block_ids.end())
        {
            throw std::runtime_error("attention_paged: slot_mapping references an unknown physical block.");
        }
    }
}

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

void run_direct_llama_attention_device(const Tensor& positions,
                                       const Tensor& q,
                                       const Tensor& k,
                                       const Tensor& v,
                                       Tensor& out,
                                       int32_t num_attention_heads,
                                       int32_t num_key_value_heads,
                                       int32_t head_dim)
{
    validate_same_device({std::cref(positions), std::cref(q), std::cref(k), std::cref(v), std::cref(out)}, "llama_attention");

    Tensor positions_cpu;
    const int32_t* positions_ptr = cpu_int_ptr(positions, positions_cpu);
    const int32_t q_hidden_size = hidden_size(num_attention_heads, head_dim);
    const int32_t kv_size = kv_hidden_size(num_key_value_heads, head_dim);
    const int32_t group_size = num_attention_heads / num_key_value_heads;
    const int64_t rows = q.size(0);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int64_t row = 0; row < rows; ++row)
    {
        const int32_t target_position = positions_ptr[row];
        for (int32_t q_head = 0; q_head < num_attention_heads; ++q_head)
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
                torch::TensorOptions().dtype(torch::kInt64).device(q.device()));
            Tensor k_context = k.index_select(0, source_index).narrow(1, kv_head * head_dim, head_dim);
            Tensor v_context = v.index_select(0, source_index).narrow(1, kv_head * head_dim, head_dim);
            Tensor q_head_tensor = q.narrow(0, row, 1).narrow(1, q_head * head_dim, head_dim);
            Tensor probs = torch::softmax(torch::matmul(q_head_tensor, k_context.transpose(0, 1)) * scale, -1);
            Tensor value = torch::matmul(probs, v_context);
            out.narrow(0, row, 1).narrow(1, q_head * head_dim, head_dim).copy_(value);
            (void)q_hidden_size;
            (void)kv_size;
        }
    }
}

void run_paged_llama_attention_device(const Tensor& positions,
                                      const Tensor& q,
                                      const Tensor& k,
                                      const Tensor& v,
                                      Tensor& out,
                                      ExecutionContext& ctx,
                                      int32_t layer_id,
                                      int32_t num_attention_heads,
                                      int32_t num_key_value_heads,
                                      int32_t head_dim)
{
    validate_same_device({std::cref(positions), std::cref(q), std::cref(k), std::cref(v), std::cref(out)}, "llama_attention");

    const PagedAttentionRuntimeMetadata& metadata = current_paged_attention_runtime_metadata();
    KVCache* kv_cache = ctx.kv();
    if (!metadata.enabled || kv_cache == nullptr)
    {
        run_direct_llama_attention_device(
            positions,
            q,
            k,
            v,
            out,
            num_attention_heads,
            num_key_value_heads,
            head_dim);
        return;
    }
    if (layer_id < 0)
    {
        throw std::runtime_error("llama_attention: layer id is not set.");
    }
    if (kv_cache->device() != q.device())
    {
        throw std::runtime_error("llama_attention: KV cache device must match attention tensors.");
    }
    if (kv_cache->block_size_tokens() != metadata.block_size_tokens)
    {
        throw std::runtime_error("llama_attention: KV block size mismatch.");
    }
    if (metadata.seq_indices == nullptr || metadata.context_lens == nullptr || metadata.block_tables == nullptr)
    {
        throw std::runtime_error("llama_attention: paged metadata is incomplete.");
    }

    const Tensor& seq_indices = *metadata.seq_indices;
    const Tensor& context_lens = *metadata.context_lens;
    const Tensor& block_tables = *metadata.block_tables;
    const std::vector<int64_t> seq_shape = tensor_shape(seq_indices);
    const std::vector<int64_t> context_shape = tensor_shape(context_lens);
    const std::vector<int64_t> block_shape = tensor_shape(block_tables);
    const int64_t rows = q.size(0);
    if (seq_shape.size() != 1 || seq_shape[0] != rows)
    {
        throw std::runtime_error("llama_attention: seq_indices shape mismatch.");
    }
    if (context_shape.size() != 1 || block_shape.size() != 3
        || block_shape[0] <= layer_id || block_shape[1] != context_shape[0])
    {
        throw std::runtime_error("llama_attention: block table shape mismatch.");
    }

    const int32_t kv_size = kv_hidden_size(num_key_value_heads, head_dim);
    const size_t kv_token_bytes = static_cast<size_t>(kv_size) * sizeof(float);
    const size_t required_block_bytes =
        2 * static_cast<size_t>(metadata.block_size_tokens) * kv_token_bytes;
    if (kv_cache->block_size_bytes() < required_block_bytes)
    {
        throw std::runtime_error("llama_attention: KV block byte size is too small.");
    }

    Tensor positions_cpu;
    Tensor seq_indices_cpu;
    Tensor context_lens_cpu;
    Tensor block_tables_cpu;
    const int32_t* positions_ptr = cpu_int_ptr(positions, positions_cpu);
    const int32_t* seq_index_ptr = cpu_int_ptr(seq_indices, seq_indices_cpu);
    const int32_t* context_ptr = cpu_int_ptr(context_lens, context_lens_cpu);
    const int32_t* block_ptr = cpu_int_ptr(block_tables, block_tables_cpu);

    const int64_t num_seqs = block_shape[1];
    const int64_t max_blocks_per_seq = block_shape[2];
    const int32_t group_size = num_attention_heads / num_key_value_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

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
            static_cast<int64_t>(layer_id) * num_seqs * max_blocks_per_seq
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
        kv_block_tensor(*kv_cache, block_id, metadata.block_size_tokens, kv_size, false)
            .narrow(0, token_offset, 1)
            .copy_(k.narrow(0, row, 1));
        kv_block_tensor(*kv_cache, block_id, metadata.block_size_tokens, kv_size, true)
            .narrow(0, token_offset, 1)
            .copy_(v.narrow(0, row, 1));
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
            torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
        Tensor v_context = torch::empty_like(k_context);
        for (int32_t src_pos = 0; src_pos <= target_position; ++src_pos)
        {
            const int32_t block_id = block_id_for(seq_index, src_pos);
            const int32_t token_offset = src_pos % metadata.block_size_tokens;
            k_context.narrow(0, src_pos, 1).copy_(
                kv_block_tensor(*kv_cache, block_id, metadata.block_size_tokens, kv_size, false)
                    .narrow(0, token_offset, 1));
            v_context.narrow(0, src_pos, 1).copy_(
                kv_block_tensor(*kv_cache, block_id, metadata.block_size_tokens, kv_size, true)
                    .narrow(0, token_offset, 1));
        }

        for (int32_t q_head = 0; q_head < num_attention_heads; ++q_head)
        {
            const int32_t kv_head = q_head / group_size;
            Tensor k_head_context = k_context.narrow(1, kv_head * head_dim, head_dim);
            Tensor v_head_context = v_context.narrow(1, kv_head * head_dim, head_dim);
            Tensor q_head_tensor = q.narrow(0, row, 1).narrow(1, q_head * head_dim, head_dim);
            Tensor probs = torch::softmax(torch::matmul(q_head_tensor, k_head_context.transpose(0, 1)) * scale, -1);
            Tensor value = torch::matmul(probs, v_head_context);
            out.narrow(0, row, 1).narrow(1, q_head * head_dim, head_dim).copy_(value);
        }
    }
}

} // namespace

#if TINYLLM_ENABLE_CUDA
namespace cuda {
void launch_attention_paged_f32(const float* q, float* out, int64_t numel, cudaStream_t stream);
} // namespace cuda

namespace {

bool is_cuda_device_accessible_pointer(const void* ptr)
{
    cudaPointerAttributes attrs{};
    const cudaError_t status = cudaPointerGetAttributes(&attrs, ptr);
    if (status != cudaSuccess)
    {
        // Host pointers are expected in CPU fallback paths under CUDA builds.
        (void)cudaGetLastError();
        return false;
    }

#if CUDART_VERSION >= 10000
    return attrs.type == cudaMemoryTypeDevice || attrs.type == cudaMemoryTypeManaged;
#else
    return attrs.memoryType == cudaMemoryTypeDevice;
#endif
}

bool can_run_cuda_attention(const float* q_ptr, const float* out_ptr)
{
    return is_cuda_device_accessible_pointer(q_ptr)
        && is_cuda_device_accessible_pointer(out_ptr);
}

} // namespace
#endif

void set_paged_attention_runtime_metadata(const Tensor& slot_mapping,
                                          const Tensor& seq_indices,
                                          const Tensor& context_lens,
                                          const Tensor& block_tables,
                                          int32_t block_size_tokens)
{
    g_runtime_metadata.slot_mapping = &slot_mapping;
    g_runtime_metadata.seq_indices = &seq_indices;
    g_runtime_metadata.context_lens = &context_lens;
    g_runtime_metadata.block_tables = &block_tables;
    g_runtime_metadata.block_size_tokens = block_size_tokens;
    g_runtime_metadata.enabled = true;
}

void clear_paged_attention_runtime_metadata()
{
    g_runtime_metadata.slot_mapping = nullptr;
    g_runtime_metadata.seq_indices = nullptr;
    g_runtime_metadata.context_lens = nullptr;
    g_runtime_metadata.block_tables = nullptr;
    g_runtime_metadata.block_size_tokens = 0;
    g_runtime_metadata.enabled = false;
}

const PagedAttentionRuntimeMetadata& current_paged_attention_runtime_metadata()
{
    return g_runtime_metadata;
}

void attention_paged(const Tensor& q, Tensor& out, ExecutionContext& ctx)
{
    if (tensor_dtype(q) != DType::kFloat32 || tensor_dtype(out) != DType::kFloat32)
    {
        throw std::runtime_error("attention_paged: only float32 tensors are supported.");
    }

    const std::vector<int64_t> q_shape = tensor_shape(q);
    const std::vector<int64_t> out_shape = tensor_shape(out);
    if (q_shape != out_shape)
    {
        throw std::runtime_error("attention_paged: q and out shapes must match.");
    }
    if (tensor_data(q) == nullptr || tensor_data(out) == nullptr)
    {
        throw std::runtime_error("attention_paged: q and out pointers must be non-null.");
    }

    validate_runtime_metadata(q);

#if TINYLLM_ENABLE_CUDA
    const float* q_ptr = static_cast<const float*>(tensor_data(q));
    float* out_ptr = static_cast<float*>(tensor_data(out));
    if (can_run_cuda_attention(q_ptr, out_ptr))
    {
        const cudaStream_t stream = resolve_execution_context(ctx).stream();
        cuda::launch_attention_paged_f32(q_ptr, out_ptr, static_cast<int64_t>(tensor_numel(q)), stream);
        return;
    }
#endif

    const size_t bytes = tensor_numel(q) * sizeof(float);
    std::memcpy(tensor_data(out), tensor_data(q), bytes);
}

void llama_attention(const Tensor& positions,
                     const Tensor& q,
                     const Tensor& k,
                     const Tensor& v,
                     Tensor& out,
                     ExecutionContext& ctx,
                     int32_t layer_id,
                     int32_t num_attention_heads,
                     int32_t num_key_value_heads,
                     int32_t head_dim)
{
    validate_llama_attention_inputs(
        positions,
        q,
        k,
        v,
        out,
        num_attention_heads,
        num_key_value_heads,
        head_dim);
    if (any_cuda({std::cref(positions), std::cref(q), std::cref(k), std::cref(v), std::cref(out)}))
    {
        run_paged_llama_attention_device(
            positions,
            q,
            k,
            v,
            out,
            ctx,
            layer_id,
            num_attention_heads,
            num_key_value_heads,
            head_dim);
        return;
    }

    validate_cpu_tensor(positions, "llama_attention");
    validate_cpu_tensor(q, "llama_attention");
    validate_cpu_tensor(k, "llama_attention");
    validate_cpu_tensor(v, "llama_attention");
    validate_cpu_tensor(out, "llama_attention");

    const int32_t* positions_ptr = static_cast<const int32_t*>(tensor_data(positions));
    const float* q_ptr = static_cast<const float*>(tensor_data(q));
    const float* k_ptr = static_cast<const float*>(tensor_data(k));
    const float* v_ptr = static_cast<const float*>(tensor_data(v));
    float* out_ptr = static_cast<float*>(tensor_data(out));

    const int32_t q_hidden_size = hidden_size(num_attention_heads, head_dim);
    const int32_t kv_size = kv_hidden_size(num_key_value_heads, head_dim);
    const int32_t group_size = num_attention_heads / num_key_value_heads;
    const int64_t rows = q.size(0);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const PagedAttentionRuntimeMetadata& metadata = current_paged_attention_runtime_metadata();
    KVCache* kv_cache = ctx.kv();
    if (metadata.enabled && kv_cache != nullptr)
    {
        if (layer_id < 0)
        {
            throw std::runtime_error("llama_attention: layer id is not set.");
        }
        if (kv_cache->block_size_tokens() != metadata.block_size_tokens)
        {
            throw std::runtime_error("llama_attention: KV block size mismatch.");
        }
        if (metadata.seq_indices == nullptr || metadata.context_lens == nullptr || metadata.block_tables == nullptr)
        {
            throw std::runtime_error("llama_attention: paged metadata is incomplete.");
        }

        const Tensor& seq_indices = *metadata.seq_indices;
        const Tensor& context_lens = *metadata.context_lens;
        const Tensor& block_tables = *metadata.block_tables;
        const std::vector<int64_t> seq_shape = tensor_shape(seq_indices);
        const std::vector<int64_t> context_shape = tensor_shape(context_lens);
        const std::vector<int64_t> block_shape = tensor_shape(block_tables);
        if (seq_shape.size() != 1 || seq_shape[0] != rows)
        {
            throw std::runtime_error("llama_attention: seq_indices shape mismatch.");
        }
        if (context_shape.size() != 1 || block_shape.size() != 3
            || block_shape[0] <= layer_id || block_shape[1] != context_shape[0])
        {
            throw std::runtime_error("llama_attention: block table shape mismatch.");
        }

        const size_t kv_token_bytes = static_cast<size_t>(kv_size) * sizeof(float);
        const size_t required_block_bytes =
            2 * static_cast<size_t>(metadata.block_size_tokens) * kv_token_bytes;
        if (kv_cache->block_size_bytes() < required_block_bytes)
        {
            throw std::runtime_error("llama_attention: KV block byte size is too small.");
        }

        const int32_t* seq_index_ptr = seq_indices.data_ptr<int32_t>();
        const int32_t* context_ptr = context_lens.data_ptr<int32_t>();
        const int32_t* block_ptr = block_tables.data_ptr<int32_t>();
        const int64_t num_layers = block_shape[0];
        const int64_t num_seqs = block_shape[1];
        const int64_t max_blocks_per_seq = block_shape[2];

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
                static_cast<int64_t>(layer_id) * num_seqs * max_blocks_per_seq
                + static_cast<int64_t>(seq_index) * max_blocks_per_seq
                + logical_block;
            (void)num_layers;
            const int32_t block_id = block_ptr[index];
            if (block_id < 0)
            {
                throw std::runtime_error("llama_attention: missing physical KV block.");
            }
            return block_id;
        };

        auto k_block_ptr = [&](int32_t block_id) -> float* {
            void* block = kv_cache->block_ptr(block_id);
            if (block == nullptr)
            {
                throw std::runtime_error("llama_attention: KV block pointer is null.");
            }
            return static_cast<float*>(block);
        };

        auto v_block_ptr = [&](int32_t block_id) -> float* {
            return k_block_ptr(block_id)
                + static_cast<size_t>(metadata.block_size_tokens) * static_cast<size_t>(kv_size);
        };

        for (int64_t row = 0; row < rows; ++row)
        {
            const int32_t seq_index = seq_index_ptr[row];
            const int32_t position = positions_ptr[row];
            const int32_t block_id = block_id_for(seq_index, position);
            const int32_t token_offset = position % metadata.block_size_tokens;
            const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(kv_size);
            float* key_dst = k_block_ptr(block_id)
                + static_cast<size_t>(token_offset) * static_cast<size_t>(kv_size);
            float* value_dst = v_block_ptr(block_id)
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
            for (int32_t q_head = 0; q_head < num_attention_heads; ++q_head)
            {
                const int32_t kv_head = q_head / group_size;
                float max_score = -std::numeric_limits<float>::infinity();
                for (int32_t src_pos = 0; src_pos <= target_position; ++src_pos)
                {
                    const int32_t block_id = block_id_for(seq_index, src_pos);
                    const int32_t token_offset = src_pos % metadata.block_size_tokens;
                    const float* key_base = k_block_ptr(block_id)
                        + static_cast<size_t>(token_offset) * static_cast<size_t>(kv_size);

                    float score = 0.0f;
                    for (int32_t dim = 0; dim < head_dim; ++dim)
                    {
                        const size_t q_index =
                            q_row_offset + static_cast<size_t>(q_head * head_dim + dim);
                        const size_t k_index =
                            static_cast<size_t>(kv_head * head_dim + dim);
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

                for (int32_t dim = 0; dim < head_dim; ++dim)
                {
                    float value = 0.0f;
                    for (int32_t src_pos = 0; src_pos <= target_position; ++src_pos)
                    {
                        const int32_t block_id = block_id_for(seq_index, src_pos);
                        const int32_t token_offset = src_pos % metadata.block_size_tokens;
                        const float* value_base = v_block_ptr(block_id)
                            + static_cast<size_t>(token_offset) * static_cast<size_t>(kv_size);
                        const size_t v_index =
                            static_cast<size_t>(kv_head * head_dim + dim);
                        value += (scores[static_cast<size_t>(src_pos)] / score_sum) * value_base[v_index];
                    }
                    const size_t out_index =
                        q_row_offset + static_cast<size_t>(q_head * head_dim + dim);
                    out_ptr[out_index] = value;
                }
            }
        }
        return;
    }

    std::vector<float> scores(static_cast<size_t>(rows), -std::numeric_limits<float>::infinity());
    for (int64_t row = 0; row < rows; ++row)
    {
        const size_t q_row_offset = static_cast<size_t>(row) * static_cast<size_t>(q_hidden_size);
        const int32_t target_position = positions_ptr[row];
        for (int32_t q_head = 0; q_head < num_attention_heads; ++q_head)
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

                const size_t k_row_offset =
                    static_cast<size_t>(src) * static_cast<size_t>(kv_size);
                float score = 0.0f;
                for (int32_t dim = 0; dim < head_dim; ++dim)
                {
                    const size_t q_index =
                        q_row_offset + static_cast<size_t>(q_head * head_dim + dim);
                    const size_t k_index =
                        k_row_offset + static_cast<size_t>(kv_head * head_dim + dim);
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

            for (int32_t dim = 0; dim < head_dim; ++dim)
            {
                float value = 0.0f;
                for (int64_t src = 0; src < rows; ++src)
                {
                    const float exp_score = scores[static_cast<size_t>(src)];
                    if (exp_score == -std::numeric_limits<float>::infinity())
                    {
                        continue;
                    }
                    const size_t v_row_offset =
                        static_cast<size_t>(src) * static_cast<size_t>(kv_size);
                    const size_t v_index =
                        v_row_offset + static_cast<size_t>(kv_head * head_dim + dim);
                    value += (exp_score / score_sum) * v_ptr[v_index];
                }
                const size_t out_index =
                    q_row_offset + static_cast<size_t>(q_head * head_dim + dim);
                out_ptr[out_index] = value;
            }
        }
    }
}

} // namespace ops
} // namespace tiny_llm
