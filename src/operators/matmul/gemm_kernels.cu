#if TINYLLM_ENABLE_CUDA
#include "utils/cuda_utils.h"

#include <cuda_runtime.h>

namespace tiny_llm::ops::cuda {

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

__global__ void gemm_f32_kernel(const float* a, const float* b, float* c, int M, int N, int K)
{
	const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
	const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
	if (row >= M || col >= N)
	{
		return;
	}

	float sum = 0.0f;
	for (int k = 0; k < K; ++k)
	{
		sum += a[static_cast<size_t>(row) * static_cast<size_t>(K) + static_cast<size_t>(k)]
			* b[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(col)];
	}

	c[static_cast<size_t>(row) * static_cast<size_t>(N) + static_cast<size_t>(col)] = sum;
}

} // namespace

void launch_gemm_f32(const float* a,
					 const float* b,
					 float* c,
					 int M,
					 int N,
					 int K,
					 cudaStream_t stream)
{
	if (M <= 0 || N <= 0 || K <= 0)
	{
		return;
	}

	const dim3 block(kBlockX, kBlockY);
	const dim3 grid(
		static_cast<unsigned int>((N + kBlockX - 1) / kBlockX),
		static_cast<unsigned int>((M + kBlockY - 1) / kBlockY));

	gemm_f32_kernel<<<grid, block, 0, stream>>>(a, b, c, M, N, K);
	CHECK_CUDA(cudaGetLastError());
}

} // namespace tiny_llm::ops::cuda
#endif
