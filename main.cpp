#include <iostream>
#include <vector>
#include <cmath>
#include "utils/cuda_utils.h"
#include "core/tensor.h"
#include "core/allocator.h"

using namespace tiny_llm;

// External CUDA launcher provided by kernels/vector_add.cu.
extern "C" void launch_vector_add(const float* a, const float* b, float* c, int n, cudaStream_t stream);

// Reference CPU implementation used to validate GPU output.
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    std::cout << "TinyLLMInference skeleton start" << std::endl;
    
    const int n = 16;
    const size_t tensor_size = n * sizeof(float);
    
    // Host buffers for inputs and CPU/GPU output comparison.
    std::vector<float> a_h(n), b_h(n), c_h_cpu(n), c_h_gpu(n);
    
    // Initialize deterministic input data.
    for (int i = 0; i < n; ++i) {
        a_h[i] = static_cast<float>(i);
        b_h[i] = static_cast<float>(i * 2);
    }
    
    // Run CPU reference path.
    vector_add_cpu(a_h.data(), b_h.data(), c_h_cpu.data(), n);
    
    // Allocate a 1 MB workspace on device memory.
    StackAllocator allocator(1024 * 1024);
    
    // Allocate tensors from the workspace allocator.
    try {
        Tensor a = allocator.make_tensor({n}, DType::kFloat32);
        Tensor b = allocator.make_tensor({n}, DType::kFloat32);
        Tensor c = allocator.make_tensor({n}, DType::kFloat32);

        float* a_dev = static_cast<float*>(a.data());
        float* b_dev = static_cast<float*>(b.data());
        float* c_dev = static_cast<float*>(c.data());
        
        // Upload input vectors to device memory.
        CHECK_CUDA(cudaMemcpy(a_dev, a_h.data(), tensor_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(b_dev, b_h.data(), tensor_size, cudaMemcpyHostToDevice));
        
        // Launch CUDA kernel on the default stream.
        launch_vector_add(a_dev, b_dev, c_dev, n, 0);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Download GPU result for verification.
        CHECK_CUDA(cudaMemcpy(c_h_gpu.data(), c_dev, tensor_size, cudaMemcpyDeviceToHost));
        
        std::cout << "vector_add completed" << std::endl;
        
        // Compare GPU output against the CPU reference.
        bool match = true;
        for (int i = 0; i < n; ++i) {
            if (std::abs(c_h_cpu[i] - c_h_gpu[i]) > 1e-6) {
                match = false;
                std::cout << "Mismatch at index " << i << ": CPU=" << c_h_cpu[i] << ", GPU=" << c_h_gpu[i] << std::endl;
            }
        }
        
        if (match) {
            std::cout << "CPU and GPU results match!" << std::endl;
        } else {
            std::cout << "CPU and GPU results do not match!" << std::endl;
        }
        
        // Rewind workspace for potential reuse in a next step.
        allocator.reset();
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
