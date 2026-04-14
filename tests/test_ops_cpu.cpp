#include <cassert>
#include <cmath>
#include <vector>

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/operators/matmul.h"
#include "tiny_llm/operators/paged_attention.h"

namespace {

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

} // namespace

int main()
{
    {
        constexpr int M = 2;
        constexpr int K = 3;
        constexpr int N = 4;
        constexpr float kTol = 1e-5f;

        const std::vector<float> a = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
        };
        const std::vector<float> b = {
            0.5f, -1.0f, 2.0f, 0.0f,
            1.5f, 0.5f, -0.5f, 1.0f,
            2.0f, 1.0f, 0.25f, -2.0f,
        };

        std::vector<float> c(M * N, 0.0f);
        std::vector<float> ref(M * N, 0.0f);

        tiny_llm::Tensor a_tensor(const_cast<float*>(a.data()), {M, K}, tiny_llm::DType::kFloat32);
        tiny_llm::Tensor b_tensor(const_cast<float*>(b.data()), {K, N}, tiny_llm::DType::kFloat32);
        tiny_llm::Tensor c_tensor(c.data(), {M, N}, tiny_llm::DType::kFloat32);
        tiny_llm::ExecutionContext ctx(nullptr, nullptr, nullptr);

        tiny_llm::ops::gemm(a_tensor, b_tensor, c_tensor, ctx);
        gemm_ref(a, b, ref, M, N, K);

        for (size_t i = 0; i < ref.size(); ++i)
        {
            assert(std::fabs(c[i] - ref[i]) < kTol);
        }
    }

    {
        constexpr int B = 2;
        constexpr int T = 3;
        constexpr int D = 4;

        std::vector<float> q(static_cast<size_t>(B * T * D), 0.0f);
        for (size_t i = 0; i < q.size(); ++i)
        {
            q[i] = static_cast<float>(i) * 0.125f - 0.5f;
        }
        std::vector<float> out(q.size(), 0.0f);

        tiny_llm::Tensor q_tensor(q.data(), {B, T, D}, tiny_llm::DType::kFloat32);
        tiny_llm::Tensor out_tensor(out.data(), {B, T, D}, tiny_llm::DType::kFloat32);
        tiny_llm::ExecutionContext ctx(nullptr, nullptr, nullptr);

        tiny_llm::ops::attention_paged(q_tensor, out_tensor, ctx);
        for (size_t i = 0; i < q.size(); ++i)
        {
            assert(std::fabs(q[i] - out[i]) < 1e-6f);
        }
    }

    return 0;
}
