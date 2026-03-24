#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime.h>

namespace tiny_llm::ops::cuda {

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void rmsnorm_f32_kernel(const float* x, const float* w, float* y, int B, int D, float eps)
{
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= B)
  {
    return;
  }

  float sum = 0.0f;
  const float* x_row = x + static_cast<size_t>(row) * static_cast<size_t>(D);
  for (int i = tid; i < D; i += blockDim.x)
  {
    const float v = x_row[i];
    sum += v * v;
  }

  __shared__ float shm[kThreadsPerBlock];
  shm[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
    {
      shm[tid] += shm[tid + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf(shm[0] / static_cast<float>(D) + eps);
  float* y_row = y + static_cast<size_t>(row) * static_cast<size_t>(D);
  for (int i = tid; i < D; i += blockDim.x)
  {
    y_row[i] = x_row[i] * inv_rms * w[i];
  }
}

} // namespace

void launch_rmsnorm_f32(
  const float* x, const float* w, float* y,
  int B, int D, float eps, cudaStream_t stream)
{
  const dim3 grid(static_cast<unsigned int>(B));
  const dim3 block(kThreadsPerBlock);
  rmsnorm_f32_kernel<<<grid, block, 0, stream>>>(x, w, y, B, D, eps);
}

} // namespace tiny_llm::ops::cuda
#endif