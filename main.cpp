#include <iostream>
#include <vector>
#include <cmath>
#include "utils/cuda_utils.h"
#include "core/tensor.h"
#include "core/allocator.h"

using namespace tiny_llm;

// declare external kernel launcher
extern "C" void launch_vector_add(const float* a, const float* b, float* c, int n, cudaStream_t stream);

// CPU version of vector add
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    std::cout << "TinyLLMInference skeleton start" << std::endl;
    
    const int n = 16;
    const size_t tensor_size = n * sizeof(float);
    
    // Host arrays for CPU computation and comparison
    std::vector<float> a_h(n), b_h(n), c_h_cpu(n), c_h_gpu(n);
    
    // Initialize host data
    for (int i = 0; i < n; ++i) {
        a_h[i] = static_cast<float>(i);
        b_h[i] = static_cast<float>(i * 2);
    }
    
    // CPU computation
    vector_add_cpu(a_h.data(), b_h.data(), c_h_cpu.data(), n);
    
    // Create StackAllocator with 1MB pool
    StackAllocator allocator(1024 * 1024);
    
    // Create tensors using allocator
    try {
        Tensor a = allocator.create_tensor({n}, DType::kFloat32);
        Tensor b = allocator.create_tensor({n}, DType::kFloat32);
        Tensor c = allocator.create_tensor({n}, DType::kFloat32);
        
        // Copy host data to GPU
        CHECK_CUDA(cudaMemcpy(a.data<float>(), a_h.data(), tensor_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(b.data<float>(), b_h.data(), tensor_size, cudaMemcpyHostToDevice));
        
        // Launch kernel
        launch_vector_add(a.data<const float>(), b.data<const float>(), 
                         c.data<float>(), n, 0);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Copy result back to host
        CHECK_CUDA(cudaMemcpy(c_h_gpu.data(), c.data<float>(), tensor_size, cudaMemcpyDeviceToHost));
        
        std::cout << "vector_add completed" << std::endl;
        
        // Compare CPU and GPU results
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
        
        // Stack allocator automatically manages memory lifecycle
        allocator.reset();
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
