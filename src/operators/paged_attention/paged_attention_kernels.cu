#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime.h>

namespace tiny_llm::ops::cuda {

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void copy_f32_kernel(const float* src, float* dst, int64_t n)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x)
        + static_cast<int64_t>(threadIdx.x);
    if (idx >= n)
    {
        return;
    }
    dst[idx] = src[idx];
}

} // namespace

void launch_attention_paged_f32(const float* q, float* out, int64_t numel, cudaStream_t stream)
{
    if (numel <= 0)
    {
        return;
    }

    const int64_t blocks = (numel + static_cast<int64_t>(kThreadsPerBlock) - 1)
        / static_cast<int64_t>(kThreadsPerBlock);

    copy_f32_kernel<<<static_cast<unsigned int>(blocks), kThreadsPerBlock, 0, stream>>>(q, out, numel);
}

} // namespace tiny_llm::ops::cuda
#endif
