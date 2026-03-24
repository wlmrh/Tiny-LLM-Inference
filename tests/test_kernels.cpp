#include <cassert>
#include <cmath>
#include <vector>

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/operators/rmsnorm.h"
#include "utils/cuda_utils.h"

namespace {

void rmsnorm_ref(const std::vector<float>& x, const std::vector<float>& w, std::vector<float>& y,
                 int B, int D, float eps)
{
    for (int b = 0; b < B; ++b)
    {
        float sum = 0.0f;
        for (int i = 0; i < D; ++i)
        {
            const float v = x[b * D + i];
            sum += v * v;
        }
        const float inv_rms = 1.0f / std::sqrt(sum / static_cast<float>(D) + eps);
        for (int i = 0; i < D; ++i)
        {
            y[b * D + i] = x[b * D + i] * inv_rms * w[i];
        }
    }
}

} // namespace

int main()
{
    constexpr int B = 2;
    constexpr int D = 8;
    constexpr float kEps = 1e-5f;
    constexpr float kTol = 1e-4f;

    const std::vector<float> h_x = {
        0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.9f, -1.1f, 0.4f,
        -0.6f, 0.8f, -0.4f, 0.2f, 1.2f, -0.9f, 0.3f, -0.5f,
    };
    const std::vector<float> h_w = {1.0f, 0.9f, 1.1f, 0.95f, 1.05f, 0.85f, 1.2f, 0.8f};

    std::vector<float> h_y(B * D, 0.0f);
    std::vector<float> h_ref(B * D, 0.0f);

    float* d_x = nullptr;
    float* d_w = nullptr;
    float* d_y = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_x), sizeof(float) * h_x.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_w), sizeof(float) * h_w.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_y), sizeof(float) * h_y.size()));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), sizeof(float) * h_x.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), sizeof(float) * h_w.size(), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));

    tiny_llm::Tensor x(d_x, {B, D}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor w(d_w, {D}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor y(d_y, {B, D}, tiny_llm::DType::kFloat32);
    tiny_llm::ExecutionContext ctx(stream, nullptr, nullptr);

    tiny_llm::ops::rmsnorm(x, w, y, ctx, kEps);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, sizeof(float) * h_y.size(), cudaMemcpyDeviceToHost));

    rmsnorm_ref(h_x, h_w, h_ref, B, D, kEps);
    for (size_t i = 0; i < h_ref.size(); ++i)
    {
        assert(std::fabs(h_y[i] - h_ref[i]) < kTol);
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
}
