#include "tiny_llm/operators/paged_attention.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/execution_context.h"

#include <cstring>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace tiny_llm {
namespace ops {

namespace {

struct PagedAttentionRuntimeMetadata {
    const Tensor* slot_mapping = nullptr;
    const Tensor* context_lens = nullptr;
    const Tensor* block_tables = nullptr;
    int32_t block_size_tokens = 0;
    bool enabled = false;
};

thread_local PagedAttentionRuntimeMetadata g_runtime_metadata;

void validate_runtime_metadata(const Tensor& q)
{
    if (!g_runtime_metadata.enabled)
    {
        return;
    }

    if (g_runtime_metadata.slot_mapping == nullptr
        || g_runtime_metadata.context_lens == nullptr
        || g_runtime_metadata.block_tables == nullptr)
    {
        throw std::runtime_error("attention_paged: runtime metadata pointers must be non-null when enabled.");
    }

    const Tensor& slot_mapping = *g_runtime_metadata.slot_mapping;
    const Tensor& context_lens = *g_runtime_metadata.context_lens;
    const Tensor& block_tables = *g_runtime_metadata.block_tables;

    if (tensor_dtype(slot_mapping) != DType::kInt32
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
    if (block_shape.size() != 2 || block_shape[0] != num_seqs)
    {
        throw std::runtime_error("attention_paged: block_tables must be rank-2 [num_seqs, max_blocks_per_seq].");
    }
    if (block_shape[1] <= 0)
    {
        throw std::runtime_error("attention_paged: block_tables second dimension must be positive.");
    }

    if (g_runtime_metadata.block_size_tokens <= 0)
    {
        throw std::runtime_error("attention_paged: block_size_tokens must be positive.");
    }

    const int32_t* slot_ptr = slot_mapping.data_ptr<int32_t>();
    const int32_t* context_ptr = context_lens.data_ptr<int32_t>();
    const int32_t* block_ptr = block_tables.data_ptr<int32_t>();

    const int64_t max_blocks_per_seq = block_shape[1];
    std::unordered_set<int32_t> known_block_ids;
    known_block_ids.reserve(static_cast<size_t>(num_seqs * max_blocks_per_seq));

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
                                          const Tensor& context_lens,
                                          const Tensor& block_tables,
                                          int32_t block_size_tokens)
{
    g_runtime_metadata.slot_mapping = &slot_mapping;
    g_runtime_metadata.context_lens = &context_lens;
    g_runtime_metadata.block_tables = &block_tables;
    g_runtime_metadata.block_size_tokens = block_size_tokens;
    g_runtime_metadata.enabled = true;
}

void clear_paged_attention_runtime_metadata()
{
    g_runtime_metadata.slot_mapping = nullptr;
    g_runtime_metadata.context_lens = nullptr;
    g_runtime_metadata.block_tables = nullptr;
    g_runtime_metadata.block_size_tokens = 0;
    g_runtime_metadata.enabled = false;
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

} // namespace ops
} // namespace tiny_llm
