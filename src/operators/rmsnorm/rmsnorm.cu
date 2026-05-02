#if TINYLLM_ENABLE_CUDA
#include "utils/cuda_utils.h"

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

// 每一个 thread_block 负责一行
__global__ void rmsnorm_f32_kernel(const float* x, const float* w, float* y, int B, int D, float eps)
{
  const int row = static_cast<int>(blockIdx.x); // 当前线程块负责的行
  const int tid = static_cast<int>(threadIdx.x); // 当前线程块属于线程块的第几个线程
  const int lane = tid % kWarpSize; // 当前的 warp
  const int warp_id = tid / kWarpSize; // 当前线程块的第几个 warp
  if (row >= B)
  {
    return;
  }

  // 计算输入、输出所在行的起始地址
  const float* x_row = x + static_cast<size_t>(row) * static_cast<size_t>(D);
  float* y_row = y + static_cast<size_t>(row) * static_cast<size_t>(D);

  __shared__ float shm_sum[kWarpsPerBlock]; // shm_sum[i] : 第 i 个 warp 中所有元素的平方和

  float thread_sum = 0.0f;
  // 一行有 D 个元素，被 blockDim.x 个元素平分
  const float4* x_rows = reinterpret_cast<const float4*>(x_row);
  for (int i = tid; i < D / 4; i += static_cast<int>(blockDim.x)){
    const float4 crt = x_rows[i];
    thread_sum += crt.x * crt.x + crt.y * crt.y + crt.z * crt.z + crt.w * crt.w;
  }

  float warp_sum = warp_reduce_sum(thread_sum);

  // 整个 warp 的和已经被加到 lane0 的 warp_sum 上
  if (lane == 0)
  {
    shm_sum[warp_id] = warp_sum;
  }
  __syncthreads();

  float block_sum = 0.0f;
  if (warp_id == 0)
  {
    block_sum = (lane < kWarpsPerBlock) ? shm_sum[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0)
    {
      shm_sum[0] = block_sum;
    }
  }
  __syncthreads();

  const float inv_rms = rsqrtf(shm_sum[0] / static_cast<float>(D) + eps);
  for (int i = tid; i < D; i += static_cast<int>(blockDim.x))
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

  // Step 2: one block handles one row, threads cooperatively process D.
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(B));
  rmsnorm_f32_kernel<<<grid, block, 0, stream>>>(x, w, y, B, D, eps);
  CHECK_CUDA(cudaGetLastError());
}

} // namespace tiny_llm::ops::cuda
#endif
