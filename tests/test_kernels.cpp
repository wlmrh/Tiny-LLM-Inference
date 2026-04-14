#include <cassert>
#include <cmath>
#include <vector>

#include "tiny_llm/core/context.h"
#include "tiny_llm/operators/matmul.h"
#include "tiny_llm/operators/paged_attention.h"
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

void gemm_ref(const std::vector<float>& a,
              const std::vector<float>& b,
              std::vector<float>& c,
              int M,
              int N,
              int K)
{
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += a[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)]
                    * b[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            }
            c[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = sum;
        }
    }
}

void test_rmsnorm(cudaStream_t stream)
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

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_y));
}

void test_gemm(cudaStream_t stream)
{
    constexpr int M = 2;
    constexpr int K = 3;
    constexpr int N = 4;
    constexpr float kTol = 1e-5f;

    const std::vector<float> h_a = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    const std::vector<float> h_b = {
        0.5f, -1.0f, 2.0f, 0.0f,
        1.5f, 0.5f, -0.5f, 1.0f,
        2.0f, 1.0f, 0.25f, -2.0f,
    };

    std::vector<float> h_c(M * N, 0.0f);
    std::vector<float> h_ref(M * N, 0.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(float) * h_a.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(float) * h_b.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_c), sizeof(float) * h_c.size()));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), sizeof(float) * h_a.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), sizeof(float) * h_b.size(), cudaMemcpyHostToDevice));

    tiny_llm::Tensor a(d_a, {M, K}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor b(d_b, {K, N}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor c(d_c, {M, N}, tiny_llm::DType::kFloat32);
    tiny_llm::ExecutionContext ctx(stream, nullptr, nullptr);

    tiny_llm::ops::gemm(a, b, c, ctx);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, sizeof(float) * h_c.size(), cudaMemcpyDeviceToHost));

    gemm_ref(h_a, h_b, h_ref, M, N, K);
    for (size_t i = 0; i < h_ref.size(); ++i)
    {
        assert(std::fabs(h_c[i] - h_ref[i]) < kTol);
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

void test_attention_paged(cudaStream_t stream)
{
    constexpr int B = 2;
    constexpr int T = 3;
    constexpr int D = 4;

    std::vector<float> h_q(static_cast<size_t>(B * T * D), 0.0f);
    for (size_t i = 0; i < h_q.size(); ++i)
    {
        h_q[i] = static_cast<float>(i) * 0.125f - 0.5f;
    }
    std::vector<float> h_out(h_q.size(), 0.0f);

    float* d_q = nullptr;
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_q), sizeof(float) * h_q.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(float) * h_out.size()));

    CHECK_CUDA(cudaMemcpy(d_q, h_q.data(), sizeof(float) * h_q.size(), cudaMemcpyHostToDevice));

    tiny_llm::Tensor q(d_q, {B, T, D}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor out(d_out, {B, T, D}, tiny_llm::DType::kFloat32);
    tiny_llm::ExecutionContext ctx(stream, nullptr, nullptr);

    tiny_llm::ops::attention_paged(q, out, ctx);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, sizeof(float) * h_out.size(), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < h_q.size(); ++i)
    {
        assert(std::fabs(h_q[i] - h_out[i]) < 1e-6f);
    }

    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_out));
}

} // namespace

int main()
{
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));

    test_rmsnorm(stream);
    test_gemm(stream);
    test_attention_paged(stream);

    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
