#include "tiny_llm/operators/rmsnorm.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/execution_context.h"

#include <limits>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tiny_llm {
namespace ops {

namespace cuda {
void launch_rmsnorm_f32(
    const float* x, const float* w, float* y,
    int B, int D, float eps, cudaStream_t stream);
} // namespace cuda

namespace {

struct RmsNormShape {
    int B;
    int D;
};

int checked_dim_to_int(int64_t dim, const char* name)
{
    if (dim <= 0)
    {
        throw std::runtime_error(std::string("rmsnorm: ") + name + " must be positive.");
    }
    if (dim > std::numeric_limits<int>::max())
    {
        throw std::runtime_error(std::string("rmsnorm: ") + name + " is too large for CUDA kernel indexing.");
    }
    return static_cast<int>(dim);
}

RmsNormShape parse_xy_shape(const Tensor& t)
{
    const std::vector<int64_t> shape = tensor_shape(t);
    if (shape.empty())
    {
        throw std::runtime_error("rmsnorm: x and y must have rank >= 1.");
    }

    const int D = checked_dim_to_int(shape.back(), "D");
    int64_t B64 = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i)
    {
        const int64_t dim = shape[i];
        if (dim <= 0)
        {
            throw std::runtime_error("rmsnorm: all dimensions in x and y must be positive.");
        }
        if (B64 > std::numeric_limits<int>::max() / dim)
        {
            throw std::runtime_error("rmsnorm: flattened batch size is too large for CUDA kernel indexing.");
        }
        B64 *= dim;
    }

    return RmsNormShape{static_cast<int>(B64), D};
}

// 输入向量 x，权重向量 w，结果写到 y，使用的 epsilon eps
void validate_rmsnorm_inputs(const Tensor& x, const Tensor& w, const Tensor& y, float eps)
{
    if (eps <= 0.0f)
    {
        throw std::runtime_error("rmsnorm: eps must be > 0.");
    }

    if (tensor_dtype(x) != DType::kFloat32 || tensor_dtype(w) != DType::kFloat32 || tensor_dtype(y) != DType::kFloat32)
    {
        throw std::runtime_error("rmsnorm: only float32 tensors are supported.");
    }

    const std::vector<int64_t> x_shape = tensor_shape(x);
    const std::vector<int64_t> w_shape = tensor_shape(w);
    const std::vector<int64_t> y_shape = tensor_shape(y);

    if (w_shape.size() != 1)
    {
        throw std::runtime_error("rmsnorm: w must be a rank-1 tensor [D].");
    }
    
    if (x_shape != y_shape)
    {
        throw std::runtime_error("rmsnorm: x and y shapes must match.");
    }

    const RmsNormShape shape = parse_xy_shape(x);

    if (w_shape[0] != static_cast<int64_t>(shape.D))
    {
        throw std::runtime_error("rmsnorm: w shape must equal D.");
    }

    if (tensor_data(x) == nullptr || tensor_data(w) == nullptr || tensor_data(y) == nullptr)
    {
        throw std::runtime_error("rmsnorm: x, w, y data pointers must be non-null.");
    }
}

bool any_cuda_tensor(const Tensor& x, const Tensor& w, const Tensor& y)
{
    return x.device().is_cuda() || w.device().is_cuda() || y.device().is_cuda();
}

void validate_same_device(const Tensor& x, const Tensor& w, const Tensor& y)
{
    if (x.device() != w.device() || x.device() != y.device())
    {
        throw std::runtime_error("rmsnorm: x, w, and y must be on the same device.");
    }
}

} // namespace

void rmsnorm(const Tensor& x, const Tensor& w, Tensor& y, ExecutionContext& ctx, float eps)
{
    validate_rmsnorm_inputs(x, w, y, eps);
    validate_same_device(x, w, y);
    const RmsNormShape shape = parse_xy_shape(x);

    const float* x_ptr = static_cast<const float*>(tensor_data(x));
    const float* w_ptr = static_cast<const float*>(tensor_data(w));
    float* y_ptr = static_cast<float*>(tensor_data(y));

#if TINYLLM_ENABLE_CUDA
    if (x.device().is_cuda())
    {
        const cudaStream_t stream = resolve_execution_context(ctx).stream();
        cuda::launch_rmsnorm_f32(
            x_ptr,
            w_ptr,
            y_ptr,
            shape.B, shape.D, eps, stream);
        return;
    }
#else
    if (any_cuda_tensor(x, w, y))
    {
        throw std::runtime_error("rmsnorm: CUDA tensors require a CUDA build.");
    }
#endif

    for (int b = 0; b < shape.B; ++b)
    {
        const size_t row_offset = static_cast<size_t>(b) * static_cast<size_t>(shape.D);
        float sum_sq = 0.0f;
        for (int d = 0; d < shape.D; ++d)
        {
            const float value = x_ptr[row_offset + static_cast<size_t>(d)];
            sum_sq += value * value;
        }

        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(shape.D) + eps);
        for (int d = 0; d < shape.D; ++d)
        {
            const size_t idx = row_offset + static_cast<size_t>(d);
            y_ptr[idx] = x_ptr[idx] * inv_rms * w_ptr[static_cast<size_t>(d)];
        }
    }
}

} // namespace ops
} // namespace tiny_llm
