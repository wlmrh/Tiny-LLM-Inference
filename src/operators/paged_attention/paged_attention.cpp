#include "tiny_llm/operators/paged_attention.h"

#include "paged_attention_internal.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/kv_cache.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace tiny_llm
{
namespace ops
{

namespace
{

bool wants_torch_reference_backend()
{
    const char *value = std::getenv("TINYLLM_PAGED_ATTENTION_BACKEND");
    return value != nullptr && std::string(value) == "torch";
}

void validate_attention_paged_metadata(const Tensor &q, const PagedAttentionRuntimeMetadata &metadata)
{
    if (!metadata.enabled)
    {
        return;
    }

    if (metadata.slot_mapping == nullptr || metadata.seq_indices == nullptr || metadata.context_lens == nullptr ||
        metadata.block_tables == nullptr)
    {
        throw std::runtime_error("attention_paged: runtime metadata pointers must be non-null when enabled.");
    }

    const Tensor &slot_mapping = *metadata.slot_mapping;
    const Tensor &seq_indices = *metadata.seq_indices;
    const Tensor &context_lens = *metadata.context_lens;
    const Tensor &block_tables = *metadata.block_tables;

    if (tensor_dtype(slot_mapping) != DType::kInt32 || tensor_dtype(seq_indices) != DType::kInt32 ||
        tensor_dtype(context_lens) != DType::kInt32 || tensor_dtype(block_tables) != DType::kInt32)
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
        throw std::runtime_error(
            "attention_paged: block_tables must be rank-3 [num_layers, num_seqs, max_blocks_per_seq].");
    }
    if (block_shape[0] <= 0 || block_shape[2] <= 0)
    {
        throw std::runtime_error("attention_paged: block_tables dimensions must be positive.");
    }
    if (metadata.block_size_tokens <= 0)
    {
        throw std::runtime_error("attention_paged: block_size_tokens must be positive.");
    }

    Tensor slot_cpu;
    Tensor seq_indices_cpu;
    Tensor context_lens_cpu;
    Tensor block_tables_cpu;
    const int32_t *slot_ptr = cpu_int_ptr(slot_mapping, slot_cpu);
    const int32_t *seq_index_ptr = cpu_int_ptr(seq_indices, seq_indices_cpu);
    const int32_t *context_ptr = cpu_int_ptr(context_lens, context_lens_cpu);
    const int32_t *block_ptr = cpu_int_ptr(block_tables, block_tables_cpu);

    const int64_t max_blocks_per_seq = block_shape[2];
    std::unordered_set<int32_t> known_block_ids;
    for (int64_t seq = 0; seq < num_seqs; ++seq)
    {
        const int32_t context_len = context_ptr[seq];
        if (context_len < 0)
        {
            throw std::runtime_error("attention_paged: context_lens values must be non-negative.");
        }
        const int64_t required_blocks =
            context_len == 0
                ? 0
                : (static_cast<int64_t>(context_len) - 1) / static_cast<int64_t>(metadata.block_size_tokens) + 1;

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

        const int32_t block_id = slot / metadata.block_size_tokens;
        if (known_block_ids.find(block_id) == known_block_ids.end())
        {
            throw std::runtime_error("attention_paged: slot_mapping references an unknown physical block.");
        }
    }
}

} // namespace

int32_t kv_hidden_size(int32_t num_key_value_heads, int32_t head_dim)
{
    return num_key_value_heads * head_dim;
}

int32_t attention_hidden_size(int32_t num_attention_heads, int32_t head_dim)
{
    return num_attention_heads * head_dim;
}

bool any_cuda(std::initializer_list<std::reference_wrapper<const Tensor>> tensors)
{
    for (const Tensor &tensor : tensors)
    {
        if (tensor.defined() && tensor.device().is_cuda())
        {
            return true;
        }
    }
    return false;
}

void validate_same_device(std::initializer_list<std::reference_wrapper<const Tensor>> tensors, const char *name)
{
    bool have_device = false;
    c10::Device device(c10::kCPU);
    for (const Tensor &tensor : tensors)
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

Tensor tensor_to_cpu_contiguous(const Tensor &tensor)
{
    if (tensor.device().is_cpu())
    {
        return tensor.contiguous();
    }
    return tensor.to(c10::kCPU, /*non_blocking=*/false, /*copy=*/true).contiguous();
}

const int32_t *cpu_int_ptr(const Tensor &tensor, Tensor &storage)
{
    storage = tensor_to_cpu_contiguous(tensor);
    return storage.data_ptr<int32_t>();
}

bool has_paged_kv_cache(const LlamaAttentionParams &params)
{
    return params.metadata != nullptr && params.metadata->enabled && params.ctx != nullptr &&
           params.ctx->kv() != nullptr;
}

void validate_llama_attention_params(const LlamaAttentionParams &params)
{
    if (params.positions == nullptr || params.q == nullptr || params.k == nullptr || params.v == nullptr ||
        params.out == nullptr || params.ctx == nullptr)
    {
        throw std::runtime_error("llama_attention: params must contain non-null tensor and context pointers.");
    }
    if (params.num_attention_heads <= 0 || params.num_key_value_heads <= 0 || params.head_dim <= 0)
    {
        throw std::runtime_error("llama_attention: attention dimensions must be positive.");
    }
    if (params.num_attention_heads % params.num_key_value_heads != 0)
    {
        throw std::runtime_error("llama_attention: num_attention_heads must be divisible by num_key_value_heads.");
    }

    const Tensor &positions = *params.positions;
    const Tensor &q = *params.q;
    const Tensor &k = *params.k;
    const Tensor &v = *params.v;
    const Tensor &out = *params.out;

    if (tensor_dtype(positions) != DType::kInt32 || tensor_dtype(q) != DType::kFloat32 ||
        tensor_dtype(k) != DType::kFloat32 || tensor_dtype(v) != DType::kFloat32 ||
        tensor_dtype(out) != DType::kFloat32)
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
    if (q.size(1) != attention_hidden_size(params.num_attention_heads, params.head_dim))
    {
        throw std::runtime_error("llama_attention: q hidden size mismatch.");
    }
    if (k.dim() != 2 || v.dim() != 2 || out.dim() != 2 || k.size(0) != rows || v.size(0) != rows ||
        out.size(0) != rows || k.size(1) != kv_hidden_size(params.num_key_value_heads, params.head_dim) ||
        v.size(1) != kv_hidden_size(params.num_key_value_heads, params.head_dim) ||
        out.size(1) != attention_hidden_size(params.num_attention_heads, params.head_dim))
    {
        throw std::runtime_error("llama_attention: k/v/out shape mismatch.");
    }
    if (tensor_data(positions) == nullptr || tensor_data(q) == nullptr || tensor_data(k) == nullptr ||
        tensor_data(v) == nullptr || tensor_data(out) == nullptr)
    {
        throw std::runtime_error("llama_attention: data pointers must be non-null.");
    }
}

void validate_paged_metadata_for_attention(const LlamaAttentionParams &params)
{
    if (!has_paged_kv_cache(params))
    {
        return;
    }
    if (params.layer_id < 0)
    {
        throw std::runtime_error("llama_attention: layer id is not set.");
    }

    KVCache *kv_cache = params.ctx->kv();
    const PagedAttentionRuntimeMetadata &metadata = *params.metadata;
    if (kv_cache->device() != params.q->device())
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
    if (metadata.block_size_tokens <= 0)
    {
        throw std::runtime_error("llama_attention: block_size_tokens must be positive.");
    }

    const Tensor &seq_indices = *metadata.seq_indices;
    const Tensor &context_lens = *metadata.context_lens;
    const Tensor &block_tables = *metadata.block_tables;
    if (tensor_dtype(seq_indices) != DType::kInt32 || tensor_dtype(context_lens) != DType::kInt32 ||
        tensor_dtype(block_tables) != DType::kInt32)
    {
        throw std::runtime_error("llama_attention: paged metadata tensors must be int32.");
    }

    const int64_t rows = params.q->size(0);
    const std::vector<int64_t> seq_shape = tensor_shape(seq_indices);
    const std::vector<int64_t> context_shape = tensor_shape(context_lens);
    const std::vector<int64_t> block_shape = tensor_shape(block_tables);
    if (seq_shape.size() != 1 || seq_shape[0] != rows)
    {
        throw std::runtime_error("llama_attention: seq_indices shape mismatch.");
    }
    if (context_shape.size() != 1 || block_shape.size() != 3 || block_shape[0] <= params.layer_id ||
        block_shape[1] != context_shape[0])
    {
        throw std::runtime_error("llama_attention: block table shape mismatch.");
    }
    if (block_shape[2] <= 0)
    {
        throw std::runtime_error("llama_attention: block table must contain at least one block column.");
    }

    const int32_t kv_size = kv_hidden_size(params.num_key_value_heads, params.head_dim);
    const size_t kv_token_bytes = static_cast<size_t>(kv_size) * runtime_dtype_size(kv_cache->dtype());
    const size_t required_block_bytes = 2 * static_cast<size_t>(metadata.block_size_tokens) * kv_token_bytes;
    if (kv_cache->block_size_bytes() < required_block_bytes)
    {
        throw std::runtime_error("llama_attention: KV block byte size is too small.");
    }
}

void attention_paged(const Tensor &q, Tensor &out, ExecutionContext &ctx)
{
#if !TINYLLM_ENABLE_CUDA
    (void)ctx;
#endif
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

    validate_attention_paged_metadata(q, PagedAttentionRuntimeMetadata{});

#if TINYLLM_ENABLE_CUDA
    const float *q_ptr = static_cast<const float *>(tensor_data(q));
    float *out_ptr = static_cast<float *>(tensor_data(out));
    if (q.device().is_cuda() && out.device().is_cuda())
    {
        namespace cuda = tiny_llm::ops::cuda;
        cuda::launch_attention_paged_f32(q_ptr, out_ptr, static_cast<int64_t>(tensor_numel(q)), ctx.stream());
        return;
    }
#endif

    const size_t bytes = tensor_numel(q) * sizeof(float);
    std::memcpy(tensor_data(out), tensor_data(q), bytes);
}

void llama_attention_forward(const LlamaAttentionParams &input_params)
{
    const LlamaAttentionParams &params = input_params;

    validate_llama_attention_params(params);
    validate_same_device({std::cref(*params.positions), std::cref(*params.q), std::cref(*params.k),
                          std::cref(*params.v), std::cref(*params.out)},
                         "llama_attention");
    validate_paged_metadata_for_attention(params);

    const bool cuda_inputs = any_cuda({std::cref(*params.positions), std::cref(*params.q), std::cref(*params.k),
                                       std::cref(*params.v), std::cref(*params.out)});

    if (!cuda_inputs)
    {
        if (has_paged_kv_cache(params))
        {
            run_paged_attention_cpu(params);
        }
        else
        {
            run_direct_attention_cpu(params);
        }
        return;
    }

    if (!wants_torch_reference_backend() && try_run_cuda_optimized_attention(params))
    {
        return;
    }

    run_torch_reference_attention(params);
}

void llama_attention(const Tensor &positions, const Tensor &q, const Tensor &k, const Tensor &v, Tensor &out,
                     ExecutionContext &ctx, int32_t layer_id, int32_t num_attention_heads, int32_t num_key_value_heads,
                     int32_t head_dim)
{
    LlamaAttentionParams params;
    params.positions = &positions;
    params.q = &q;
    params.k = &k;
    params.v = &v;
    params.out = &out;
    params.ctx = &ctx;
    params.metadata = nullptr;
    params.layer_id = layer_id;
    params.num_attention_heads = num_attention_heads;
    params.num_key_value_heads = num_key_value_heads;
    params.head_dim = head_dim;
    llama_attention_forward(params);
}

} // namespace ops
} // namespace tiny_llm
