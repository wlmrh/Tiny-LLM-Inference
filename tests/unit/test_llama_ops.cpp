#include "tiny_llm/operators/llama_ops.h"

#include <cmath>
#include <gtest/gtest.h>

TEST(LlamaOpsTest, EmbeddingLookupSupportsBothLayouts)
{
    tiny_llm::Tensor ids = torch::tensor({2, 0}, torch::TensorOptions().dtype(torch::kInt32));
    tiny_llm::Tensor embedding = torch::tensor(
        {{1.0f, 2.0f},
         {3.0f, 4.0f},
         {5.0f, 6.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor embedded = torch::empty({2, 2}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::ops::embedding_lookup(ids, embedding, embedded, 3, 2, true);
    EXPECT_NEAR(embedded.data_ptr<float>()[0], 5.0f, 1e-5f);
    EXPECT_NEAR(embedded.data_ptr<float>()[1], 6.0f, 1e-5f);
    EXPECT_NEAR(embedded.data_ptr<float>()[2], 1.0f, 1e-5f);
    EXPECT_NEAR(embedded.data_ptr<float>()[3], 2.0f, 1e-5f);

    tiny_llm::Tensor transposed_embedding = embedding.transpose(0, 1).contiguous();
    tiny_llm::Tensor embedded_transposed = torch::empty({2, 2}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::ops::embedding_lookup(ids, transposed_embedding, embedded_transposed, 3, 2, false);
    EXPECT_NEAR(embedded_transposed.data_ptr<float>()[0], 5.0f, 1e-5f);
    EXPECT_NEAR(embedded_transposed.data_ptr<float>()[3], 2.0f, 1e-5f);
}

TEST(LlamaOpsTest, SplitsQkvAndAppliesRope)
{
    tiny_llm::Tensor qkv = torch::tensor(
        {{1.0f, 2.0f, 3.0f, 4.0f},
         {5.0f, 6.0f, 7.0f, 8.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor q = torch::empty({2, 2}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor k = torch::empty({2, 1}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor v = torch::empty({2, 1}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::ops::split_qkv(qkv, q, k, v, 2, 1);
    EXPECT_NEAR(q.data_ptr<float>()[0], 1.0f, 1e-5f);
    EXPECT_NEAR(k.data_ptr<float>()[1], 7.0f, 1e-5f);
    EXPECT_NEAR(v.data_ptr<float>()[1], 8.0f, 1e-5f);

    tiny_llm::Tensor positions = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32));
    tiny_llm::Tensor rope_q = torch::tensor({{1.0f, 2.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor rope_k = torch::tensor({{3.0f, 4.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::ops::apply_rope(positions, rope_q, rope_k, 1, 1, 2, 10000.0f);
    const float c = std::cos(1.0f);
    const float s = std::sin(1.0f);
    EXPECT_NEAR(rope_q.data_ptr<float>()[0], 1.0f * c - 2.0f * s, 1e-5f);
    EXPECT_NEAR(rope_q.data_ptr<float>()[1], 2.0f * c + 1.0f * s, 1e-5f);
    EXPECT_NEAR(rope_k.data_ptr<float>()[0], 3.0f * c - 4.0f * s, 1e-5f);
    EXPECT_NEAR(rope_k.data_ptr<float>()[1], 4.0f * c + 3.0f * s, 1e-5f);
}

TEST(LlamaOpsTest, AppliesLlama3RopeScaling)
{
    tiny_llm::Tensor positions = torch::tensor({8192}, torch::TensorOptions().dtype(torch::kInt32));
    tiny_llm::Tensor q = torch::tensor({{0.0f, 1.0f, 2.0f, 3.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor k = torch::tensor({{4.0f, 5.0f, 6.0f, 7.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::ops::apply_rope(positions, q, k, 1, 1, 4, 500000.0f, "llama3", 32.0f, 1.0f, 4.0f, 8192);

    const float base_inv_freq = 1.0f / std::sqrt(500000.0f);
    const float wavelen = (2.0f * static_cast<float>(M_PI)) / base_inv_freq;
    const float smooth_factor = (8192.0f / wavelen - 1.0f) / (4.0f - 1.0f);
    const float inv_freq = (1.0f - smooth_factor) * (base_inv_freq / 32.0f) + smooth_factor * base_inv_freq;
    const float theta = 8192.0f * inv_freq;
    const float c = std::cos(theta);
    const float s = std::sin(theta);
    EXPECT_NEAR(q.data_ptr<float>()[1], 1.0f * c - 3.0f * s, 1e-5f);
    EXPECT_NEAR(q.data_ptr<float>()[3], 3.0f * c + 1.0f * s, 1e-5f);
    EXPECT_NEAR(k.data_ptr<float>()[1], 5.0f * c - 7.0f * s, 1e-5f);
    EXPECT_NEAR(k.data_ptr<float>()[3], 7.0f * c + 5.0f * s, 1e-5f);
}

#if TINYLLM_ENABLE_CUDA
TEST(LlamaOpsTest, CudaRopeCacheMatchesInvFreqPath)
{
    if (!torch::cuda::is_available())
    {
        GTEST_SKIP() << "CUDA is not available.";
    }

    tiny_llm::Tensor positions = torch::tensor(
        {0, 1, 17, 4097, 5000},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    tiny_llm::Tensor q_base = torch::arange(
        120,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({5, 24}) / 17.0f;
    tiny_llm::Tensor k_base = torch::arange(
        80,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({5, 16}) / 19.0f;
    tiny_llm::Tensor q_ref = q_base.clone();
    tiny_llm::Tensor k_ref = k_base.clone();
    tiny_llm::Tensor q_cached = q_base.clone();
    tiny_llm::Tensor k_cached = k_base.clone();

    tiny_llm::Tensor inv_freq = torch::tensor(
        {1.0f, 0.1f, 0.01f, 0.001f},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    tiny_llm::Tensor cache_positions = torch::arange(
        5001,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).unsqueeze(1);
    tiny_llm::Tensor theta = cache_positions * inv_freq.unsqueeze(0);
    tiny_llm::Tensor cos_cache = torch::cos(theta).contiguous();
    tiny_llm::Tensor sin_cache = torch::sin(theta).contiguous();

    tiny_llm::ops::apply_rope(positions, q_ref, k_ref, 3, 2, 8, inv_freq);
    tiny_llm::ops::apply_rope(positions, q_cached, k_cached, 3, 2, 8, cos_cache, sin_cache);
    torch::cuda::synchronize();

    tiny_llm::Tensor q_diff = (q_ref - q_cached).abs().max().cpu();
    tiny_llm::Tensor k_diff = (k_ref - k_cached).abs().max().cpu();
    EXPECT_LT(q_diff.item<float>(), 1e-6f);
    EXPECT_LT(k_diff.item<float>(), 1e-6f);
}
#endif

TEST(LlamaOpsTest, ElementwiseHelpersMatchExpectedValues)
{
    tiny_llm::Tensor gate = torch::tensor({{-1.0f, 0.0f, 1.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor up = torch::tensor({{2.0f, 3.0f, 4.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor activated = torch::empty({1, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::ops::silu_multiply(gate, up, activated);
    EXPECT_NEAR(activated.data_ptr<float>()[0], (-1.0f / (1.0f + std::exp(1.0f))) * 2.0f, 1e-5f);
    EXPECT_NEAR(activated.data_ptr<float>()[2], (1.0f / (1.0f + std::exp(-1.0f))) * 4.0f, 1e-5f);

    tiny_llm::Tensor copied = torch::empty_like(activated);
    tiny_llm::ops::copy_tensor(activated, copied);
    EXPECT_NEAR(copied.data_ptr<float>()[2], activated.data_ptr<float>()[2], 1e-5f);

    tiny_llm::Tensor added = torch::empty_like(activated);
    tiny_llm::ops::add_tensors(activated, copied, added);
    EXPECT_NEAR(added.data_ptr<float>()[2], activated.data_ptr<float>()[2] * 2.0f, 1e-5f);
}

#if TINYLLM_ENABLE_CUDA
TEST(LlamaOpsTest, CudaSiluMultiplyMatchesCpu)
{
    if (!torch::cuda::is_available())
    {
        GTEST_SKIP() << "CUDA is not available.";
    }
    tiny_llm::Tensor gate_cpu = torch::tensor({{-2.0f, -0.25f, 0.0f, 1.0f, 3.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor up_cpu = torch::tensor({{1.5f, -2.0f, 3.0f, 0.5f, -1.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor expected = torch::empty_like(gate_cpu);
    tiny_llm::ops::silu_multiply(gate_cpu, up_cpu, expected);

    tiny_llm::Tensor actual = torch::empty_like(gate_cpu).to(torch::kCUDA);
    tiny_llm::ops::silu_multiply(gate_cpu.to(torch::kCUDA), up_cpu.to(torch::kCUDA), actual);
    torch::cuda::synchronize();
    EXPECT_LT((actual.cpu() - expected).abs().max().item<float>(), 1e-6f);
}
#endif
