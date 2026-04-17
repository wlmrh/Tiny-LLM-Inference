#include "tiny_llm/operators/paged_attention.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/execution_context.h"

#include <cstring>
#include <stdexcept>

#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace tiny_llm {
namespace ops {

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

void attention_paged(const Tensor& q, Tensor& out, ExecutionContext& ctx)
{
    if (q.dtype() != DType::kFloat32 || out.dtype() != DType::kFloat32)
    {
        throw std::runtime_error("attention_paged: only float32 tensors are supported.");
    }
    if (q.shape() != out.shape())
    {
        throw std::runtime_error("attention_paged: q and out shapes must match.");
    }
    if (q.data() == nullptr || out.data() == nullptr)
    {
        throw std::runtime_error("attention_paged: q and out pointers must be non-null.");
    }

#if TINYLLM_ENABLE_CUDA
    const float* q_ptr = static_cast<const float*>(q.data());
    float* out_ptr = static_cast<float*>(out.data());
    if (can_run_cuda_attention(q_ptr, out_ptr))
    {
        const cudaStream_t stream = resolve_execution_context(ctx).stream();
        cuda::launch_attention_paged_f32(q_ptr, out_ptr, static_cast<int64_t>(q.numel()), stream);
        return;
    }
#endif

    const size_t bytes = q.numel() * sizeof(float);
    std::memcpy(out.data(), q.data(), bytes);
}

} // namespace ops
} // namespace tiny_llm
