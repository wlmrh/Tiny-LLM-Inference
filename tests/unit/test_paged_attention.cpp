#include "tiny_llm/core/allocator.h"
#include "tiny_llm/core/context.h"
#include "tiny_llm/operators/paged_attention.h"
#include "tiny_llm/runtime/kv_cache.h"
#include "tiny_llm/runtime/parallel_config.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void expect_near(float actual, float expected, const char* message)
{
    if (std::fabs(actual - expected) > 1e-5f)
    {
        throw std::runtime_error(message);
    }
}

std::vector<float> expected_two_token_attention()
{
    const float scale = 1.0f / std::sqrt(2.0f);
    const float s0 = std::exp(0.0f - scale);
    const float s1 = 1.0f;
    const float denom = s0 + s1;
    return {
        10.0f,
        20.0f,
        (s0 * 10.0f + s1 * 30.0f) / denom,
        (s0 * 20.0f + s1 * 40.0f) / denom,
    };
}

void check_output(const tiny_llm::Tensor& out)
{
    const std::vector<float> expected = expected_two_token_attention();
    const float* ptr = out.data_ptr<float>();
    for (size_t i = 0; i < expected.size(); ++i)
    {
        expect_near(ptr[i], expected[i], "attention output mismatch.");
    }
}

} // namespace

int main()
{
    tiny_llm::Tensor positions = torch::tensor({0, 1}, torch::TensorOptions().dtype(torch::kInt32));
    tiny_llm::Tensor q = torch::tensor(
        {{1.0f, 0.0f},
         {0.0f, 1.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor k = q.clone();
    tiny_llm::Tensor v = torch::tensor(
        {{10.0f, 20.0f},
         {30.0f, 40.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));

    tiny_llm::ExecutionContext direct_ctx(nullptr, nullptr, nullptr);
    tiny_llm::Tensor direct_out = torch::empty({2, 2}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::ops::llama_attention(
        positions,
        q,
        k,
        v,
        direct_out,
        direct_ctx,
        -1,
        1,
        1,
        2);
    check_output(direct_out);

    constexpr int32_t kBlockSizeTokens = 2;
    constexpr int32_t kKvHiddenSize = 2;
    constexpr size_t kBlockFloats = 2 * kBlockSizeTokens * kKvHiddenSize;
    std::vector<float> kv_pool(kBlockFloats, 0.0f);
    tiny_llm::BlockAllocator blocks(
        1,
        kBlockFloats * sizeof(float),
        kv_pool.data(),
        tiny_llm::ParallelConfig::cpu());
    tiny_llm::KVCache::Config kv_cfg;
    kv_cfg.num_layers = 1;
    kv_cfg.block_size_tokens = kBlockSizeTokens;
    tiny_llm::KVCache kv_cache(kv_cfg, &blocks);
    tiny_llm::ExecutionContext paged_ctx(nullptr, nullptr, &kv_cache);

    tiny_llm::Tensor slot_mapping = torch::tensor({0, 1}, torch::TensorOptions().dtype(torch::kInt32));
    tiny_llm::Tensor seq_indices = torch::tensor({0, 0}, torch::TensorOptions().dtype(torch::kInt32));
    tiny_llm::Tensor context_lens = torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt32));
    tiny_llm::Tensor block_tables = torch::tensor({{{0}}}, torch::TensorOptions().dtype(torch::kInt32));
    tiny_llm::ops::set_paged_attention_runtime_metadata(
        slot_mapping,
        seq_indices,
        context_lens,
        block_tables,
        kBlockSizeTokens);

    tiny_llm::Tensor paged_out = torch::empty({2, 2}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::ops::llama_attention(
        positions,
        q,
        k,
        v,
        paged_out,
        paged_ctx,
        0,
        1,
        1,
        2);
    tiny_llm::ops::clear_paged_attention_runtime_metadata();
    check_output(paged_out);

    const float* kv_ptr = kv_pool.data();
    expect_near(kv_ptr[0], 1.0f, "cached key token 0 dim 0 mismatch.");
    expect_near(kv_ptr[3], 1.0f, "cached key token 1 dim 1 mismatch.");
    expect_near(kv_ptr[4], 10.0f, "cached value token 0 dim 0 mismatch.");
    expect_near(kv_ptr[7], 40.0f, "cached value token 1 dim 1 mismatch.");

    return 0;
}
