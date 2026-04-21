#include "tiny_llm/operators/matmul.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/execution_context.h"

#include <limits>
#include <stdexcept>
#include <string>

#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

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
    if (tensor_dtype(a) != DType::kFloat32 || tensor_dtype(b) != DType::kFloat32 || tensor_dtype(c) != DType::kFloat32)
    {
        throw std::runtime_error("gemm: only float32 tensors are supported.");
    }

    const std::vector<int64_t> a_shape = tensor_shape(a);
    const std::vector<int64_t> b_shape = tensor_shape(b);
    const std::vector<int64_t> c_shape = tensor_shape(c);

    if (a_shape.size() != 2 || b_shape.size() != 2 || c_shape.size() != 2)
    {
        throw std::runtime_error("gemm: a, b, c must all be rank-2 tensors.");
    }

    const int64_t M = a_shape[0];
    const int64_t K = a_shape[1];
    const int64_t Kb = b_shape[0];
    const int64_t N = b_shape[1];

    if (K != Kb)
    {
        throw std::runtime_error("gemm: incompatible inner dimensions between a and b.");
    }
    if (c_shape[0] != M || c_shape[1] != N)
    {
        throw std::runtime_error("gemm: c shape must be [a.rows, b.cols].");
    }
    if (tensor_data(a) == nullptr || tensor_data(b) == nullptr || tensor_data(c) == nullptr)
    {
        throw std::runtime_error("gemm: input/output data pointers must be non-null.");
    }
}

void run_gemm_cpu(const float* a_ptr,
                  const float* b_ptr,
                  float* c_ptr,
                  int M,
                  int N,
                  int K)
{
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
}

#if TINYLLM_ENABLE_CUDA
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

bool can_run_cuda_gemm(const float* a_ptr, const float* b_ptr, const float* c_ptr)
{
    return is_cuda_device_accessible_pointer(a_ptr)
        && is_cuda_device_accessible_pointer(b_ptr)
        && is_cuda_device_accessible_pointer(c_ptr);
}
#endif

} // namespace

void gemm(const Tensor& a, const Tensor& b, Tensor& c, ExecutionContext& ctx)
{
    validate_gemm_inputs(a, b, c);

    const std::vector<int64_t> a_shape = tensor_shape(a);
    const std::vector<int64_t> b_shape = tensor_shape(b);
    const int M = checked_dim_to_int64(a_shape[0], "M");
    const int K = checked_dim_to_int64(a_shape[1], "K");
    const int N = checked_dim_to_int64(b_shape[1], "N");

    const float* a_ptr = static_cast<const float*>(tensor_data(a));
    const float* b_ptr = static_cast<const float*>(tensor_data(b));
    float* c_ptr = static_cast<float*>(tensor_data(c));

#if TINYLLM_ENABLE_CUDA
    if (can_run_cuda_gemm(a_ptr, b_ptr, c_ptr))
    {
        const cudaStream_t stream = resolve_execution_context(ctx).stream();
        cuda::launch_gemm_f32(a_ptr, b_ptr, c_ptr, M, N, K, stream);
        return;
    }
#endif

    run_gemm_cpu(a_ptr, b_ptr, c_ptr, M, N, K);
}

} // namespace ops
} // namespace tiny_llm
