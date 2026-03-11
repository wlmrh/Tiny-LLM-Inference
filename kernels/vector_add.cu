#include <cuda_runtime.h>

// simple vector addition kernel for first-week smoke test
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" void launch_vector_add(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vector_add<<<gridSize, blockSize, 0, stream>>>(a, b, c, n);
}
