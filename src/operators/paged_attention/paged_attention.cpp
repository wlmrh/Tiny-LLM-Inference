#include "tiny_llm/operators/paged_attention.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/execution_context.h"

#include <cstring>
#include <stdexcept>

namespace tiny_llm {
namespace ops {

#if TINYLLM_ENABLE_CUDA
namespace cuda {
void launch_attention_paged_f32(const float* q, float* out, int64_t numel, cudaStream_t stream);
} // namespace cuda
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
    const cudaStream_t stream = resolve_execution_context(ctx).stream();
    cuda::launch_attention_paged_f32(
        static_cast<const float*>(q.data()),
        static_cast<float*>(out.data()),
        static_cast<int64_t>(q.numel()),
        stream);
#else
    const size_t bytes = q.numel() * sizeof(float);
    std::memcpy(out.data(), q.data(), bytes);
#endif
}

} // namespace ops
} // namespace tiny_llm
