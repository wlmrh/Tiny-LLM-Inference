#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime.h>

namespace tiny_llm::ops::cuda {

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = (kThreadsPerBlock + kWarpSize - 1) / kWarpSize;

static_assert(kThreadsPerBlock % kWarpSize == 0, "kThreadsPerBlock must be a multiple of warp size.");

__device__ __forceinline__ float warp_reduce_sum(float val)
{
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
  {
    val += __shfl_down_sync(0xffffffffu, val, offset);
  }
  return val;
}

__global__ void rmsnorm_f32_kernel(const float* x, const float* w, float* y, int B, int D, float eps)
{
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % kWarpSize;
  const int warp_id = tid / kWarpSize;
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

  const float warp_sum = warp_reduce_sum(sum);

  __shared__ float shm[kWarpsPerBlock];
  if (lane == 0)
  {
    shm[warp_id] = warp_sum;
  }
  __syncthreads();

  if (warp_id == 0)
  {
    float block_sum = (lane < kWarpsPerBlock) ? shm[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0)
    {
      shm[0] = block_sum;
    }
  }
  __syncthreads();

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
  if (B <= 0 || D <= 0)
  {
    return;
  }
  const dim3 grid(static_cast<unsigned int>(B));
  const dim3 block(kThreadsPerBlock);
  rmsnorm_f32_kernel<<<grid, block, 0, stream>>>(x, w, y, B, D, eps);
}

} // namespace tiny_llm::ops::cuda
#endif