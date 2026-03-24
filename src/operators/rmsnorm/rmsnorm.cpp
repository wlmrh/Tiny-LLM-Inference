#include "tiny_llm/operators/rmsnorm.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"

#include <stdexcept>

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

RmsNormShape parse_xy_shape(const Tensor& t)
{
    if (t.shape().size() == 1)
    {
        if (t.shape()[0] <= 0)
        {
            throw std::runtime_error("rmsnorm: D must be positive.");
        }
        return RmsNormShape{1, static_cast<int>(t.shape()[0])};
    }
    if (t.shape().size() == 2)
    {
        if (t.shape()[0] <= 0 || t.shape()[1] <= 0)
        {
            throw std::runtime_error("rmsnorm: B and D must be positive.");
        }
        return RmsNormShape{static_cast<int>(t.shape()[0]), static_cast<int>(t.shape()[1])};
    }
    throw std::runtime_error("rmsnorm: x and y must be rank-1 [D] or rank-2 [B, D].");
}

void validate_rmsnorm_inputs(const Tensor& x, const Tensor& w, const Tensor& y, float eps)
{
    if (eps <= 0.0f)
    {
        throw std::runtime_error("rmsnorm: eps must be > 0.");
    }
    if (x.dtype() != DType::kFloat32 || w.dtype() != DType::kFloat32 || y.dtype() != DType::kFloat32)
    {
        throw std::runtime_error("rmsnorm: only float32 tensors are supported.");
    }
    if (w.shape().size() != 1)
    {
        throw std::runtime_error("rmsnorm: w must be a rank-1 tensor [D].");
    }
    if (x.shape() != y.shape())
    {
        throw std::runtime_error("rmsnorm: x and y shapes must match.");
    }
    const RmsNormShape shape = parse_xy_shape(x);
    if (w.shape()[0] != shape.D)
    {
        throw std::runtime_error("rmsnorm: w shape must equal D.");
    }
    if (x.data() == nullptr || w.data() == nullptr || y.data() == nullptr)
    {
        throw std::runtime_error("rmsnorm: x, w, y data pointers must be non-null.");
    }
}

} // namespace

void rmsnorm(const Tensor& x, const Tensor& w, Tensor& y, ExecutionContext& ctx, float eps)
{
    validate_rmsnorm_inputs(x, w, y, eps);
    const RmsNormShape shape = parse_xy_shape(x);

#if TINYLLM_ENABLE_CUDA
    cuda::launch_rmsnorm_f32(
        static_cast<const float*>(x.data()),
        static_cast<const float*>(w.data()),
        static_cast<float*>(y.data()),
        shape.B, shape.D, eps, ctx.stream());
#else
    (void)x;
    (void)w;
    (void)y;
    (void)ctx;
    (void)eps;
    throw std::runtime_error("rmsnorm CPU backend is not implemented. Rebuild with -DTINYLLM_ENABLE_CUDA=ON.");
#endif
}

} // namespace ops
} // namespace tiny_llm
