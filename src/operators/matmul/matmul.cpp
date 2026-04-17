#include "tiny_llm/operators/matmul.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/execution_context.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace tiny_llm {
namespace ops {

#if TINYLLM_ENABLE_CUDA
namespace cuda {
void launch_gemm_f32(const float* a,
                     const float* b,
                     float* c,
                     int M,
                     int N,
                     int K,
                     cudaStream_t stream);
} // namespace cuda
#endif

namespace {

int checked_dim_to_int64(int64_t dim, const char* name)
{
    if (dim <= 0)
    {
        throw std::runtime_error(std::string("gemm: ") + name + " must be positive.");
    }
    if (dim > std::numeric_limits<int>::max())
    {
        throw std::runtime_error(std::string("gemm: ") + name + " is too large for indexing.");
    }
    return static_cast<int>(dim);
}

void validate_gemm_inputs(const Tensor& a, const Tensor& b, const Tensor& c)
{
    if (a.dtype() != DType::kFloat32 || b.dtype() != DType::kFloat32 || c.dtype() != DType::kFloat32)
    {
        throw std::runtime_error("gemm: only float32 tensors are supported.");
    }
    if (a.shape().size() != 2 || b.shape().size() != 2 || c.shape().size() != 2)
    {
        throw std::runtime_error("gemm: a, b, c must all be rank-2 tensors.");
    }

    const int64_t M = a.shape()[0];
    const int64_t K = a.shape()[1];
    const int64_t Kb = b.shape()[0];
    const int64_t N = b.shape()[1];

    if (K != Kb)
    {
        throw std::runtime_error("gemm: incompatible inner dimensions between a and b.");
    }
    if (c.shape()[0] != M || c.shape()[1] != N)
    {
        throw std::runtime_error("gemm: c shape must be [a.rows, b.cols].");
    }
    if (a.data() == nullptr || b.data() == nullptr || c.data() == nullptr)
    {
        throw std::runtime_error("gemm: input/output data pointers must be non-null.");
    }
}

} // namespace

void gemm(const Tensor& a, const Tensor& b, Tensor& c, ExecutionContext& ctx)
{
    validate_gemm_inputs(a, b, c);

    const int M = checked_dim_to_int64(a.shape()[0], "M");
    const int K = checked_dim_to_int64(a.shape()[1], "K");
    const int N = checked_dim_to_int64(b.shape()[1], "N");

    const float* a_ptr = static_cast<const float*>(a.data());
    const float* b_ptr = static_cast<const float*>(b.data());
    float* c_ptr = static_cast<float*>(c.data());

#if TINYLLM_ENABLE_CUDA
    const cudaStream_t stream = resolve_execution_context(ctx).stream();
    cuda::launch_gemm_f32(a_ptr, b_ptr, c_ptr, M, N, K, stream);
#else
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += a_ptr[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)]
                    * b_ptr[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            }
            c_ptr[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = sum;
        }
    }
#endif
}

} // namespace ops
} // namespace tiny_llm
