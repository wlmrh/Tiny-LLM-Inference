#include "tiny_llm/core/allocator.h"
#include "tiny_llm/core/context.h"
#include "tiny_llm/operators/paged_attention.h"
#include "tiny_llm/runtime/kv_cache.h"
#include "tiny_llm/runtime/parallel_config.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void check_cuda(cudaError_t status, const char *message)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

void expect_close(const tiny_llm::Tensor &actual, const tiny_llm::Tensor &expected, float tolerance)
{
    tiny_llm::Tensor a = actual.cpu().contiguous();
    tiny_llm::Tensor e = expected.cpu().contiguous();
    ASSERT_EQ(a.numel(), e.numel());
    const float *ap = a.data_ptr<float>();
    const float *ep = e.data_ptr<float>();
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        EXPECT_NEAR(ap[i], ep[i], tolerance) << "index " << i;
    }
}

tiny_llm::Tensor int_cuda(std::vector<int32_t> values, std::vector<int64_t> shape)
{
    return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32)).reshape(shape).to(torch::kCUDA);
}

tiny_llm::Tensor make_values(int64_t rows, int64_t cols, float offset)
{
    tiny_llm::Tensor values =
        torch::arange(rows * cols, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    return torch::sin((values.reshape({rows, cols}) + offset) / 17.0f);
}

std::vector<int32_t> positions(int32_t start, int32_t count)
{
    std::vector<int32_t> values;
    values.reserve(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i)
    {
        values.push_back(start + i);
    }
    return values;
}

std::vector<int32_t> repeated(int32_t value, int32_t count)
{
    return std::vector<int32_t>(static_cast<size_t>(count), value);
}

std::vector<int32_t> block_table(int32_t block_count)
{
    std::vector<int32_t> values;
    values.reserve(static_cast<size_t>(block_count));
    for (int32_t i = 0; i < block_count; ++i)
    {
        values.push_back(i);
    }
    return values;
}

struct CaseResult
{
    tiny_llm::Tensor prefill;
    tiny_llm::Tensor decode;
};

CaseResult run_prefill_then_decode(bool optimized, int32_t num_attention_heads, int32_t num_key_value_heads,
                                   int32_t head_dim, int32_t block_size_tokens, int32_t prefill_tokens,
                                   tiny_llm::RuntimeDType kv_dtype = tiny_llm::RuntimeDType::kFloat32)
{
    if (optimized)
    {
        setenv("TINYLLM_PAGED_ATTENTION_BACKEND", "cuda", 1);
    }
    else
    {
        setenv("TINYLLM_PAGED_ATTENTION_BACKEND", "torch", 1);
    }

    const int32_t block_count = (prefill_tokens + 1 + block_size_tokens - 1) / block_size_tokens;
    const int32_t kv_size = num_key_value_heads * head_dim;
    const int32_t hidden_size = num_attention_heads * head_dim;
    const size_t block_elements = 2 * static_cast<size_t>(block_size_tokens) * static_cast<size_t>(kv_size);
    const c10::ScalarType pool_type =
        kv_dtype == tiny_llm::RuntimeDType::kBFloat16 ? torch::kBFloat16 : torch::kFloat32;
    tiny_llm::Tensor pool = torch::zeros({static_cast<int64_t>(block_count * block_elements)},
                                         torch::TensorOptions().dtype(pool_type).device(torch::kCUDA));

    tiny_llm::BlockAllocator blocks(block_count, block_elements * tiny_llm::runtime_dtype_size(kv_dtype),
                                    pool.data_ptr(), tiny_llm::ParallelConfig::cuda(0));
    tiny_llm::KVCache::Config kv_cfg;
    kv_cfg.num_layers = 1;
    kv_cfg.block_size_tokens = block_size_tokens;
    kv_cfg.dtype = kv_dtype;
    tiny_llm::KVCache kv_cache(kv_cfg, &blocks);
    tiny_llm::ExecutionContext ctx(nullptr, nullptr, &kv_cache, tiny_llm::ParallelConfig::cuda(0));

    tiny_llm::Tensor q_prefill = make_values(prefill_tokens, hidden_size, 1.0f);
    tiny_llm::Tensor k_prefill = make_values(prefill_tokens, kv_size, 11.0f);
    tiny_llm::Tensor v_prefill = make_values(prefill_tokens, kv_size, 21.0f);
    tiny_llm::Tensor out_prefill = torch::empty_like(q_prefill);
    tiny_llm::Tensor positions_prefill = int_cuda(positions(0, prefill_tokens), {prefill_tokens});
    tiny_llm::Tensor seq_indices_prefill = int_cuda(repeated(0, prefill_tokens), {prefill_tokens});
    tiny_llm::Tensor context_prefill = int_cuda({prefill_tokens}, {1});
    tiny_llm::Tensor blocks_prefill = int_cuda(block_table(block_count), {1, 1, block_count});
    std::vector<tiny_llm::ops::PagedAttentionPrefillSegment> prefill_segments;
    if (optimized)
    {
        prefill_segments.push_back({0, 0, prefill_tokens});
    }
    tiny_llm::ops::PagedAttentionRuntimeMetadata prefill_metadata;
    prefill_metadata.seq_indices = &seq_indices_prefill;
    prefill_metadata.context_lens = &context_prefill;
    prefill_metadata.block_tables = &blocks_prefill;
    prefill_metadata.prefill_segments = prefill_segments.empty() ? nullptr : prefill_segments.data();
    prefill_metadata.prefill_segment_count = static_cast<int64_t>(prefill_segments.size());
    prefill_metadata.block_size_tokens = block_size_tokens;
    prefill_metadata.prefill_segments_valid = !prefill_segments.empty();
    prefill_metadata.enabled = true;
    tiny_llm::ops::LlamaAttentionParams prefill;
    prefill.positions = &positions_prefill;
    prefill.q = &q_prefill;
    prefill.k = &k_prefill;
    prefill.v = &v_prefill;
    prefill.out = &out_prefill;
    prefill.ctx = &ctx;
    prefill.metadata = &prefill_metadata;
    prefill.layer_id = 0;
    prefill.num_attention_heads = num_attention_heads;
    prefill.num_key_value_heads = num_key_value_heads;
    prefill.head_dim = head_dim;
    tiny_llm::ops::llama_attention_forward(prefill);
    check_cuda(cudaDeviceSynchronize(), "prefill synchronize");

    tiny_llm::Tensor q_decode = make_values(1, hidden_size, 31.0f);
    tiny_llm::Tensor k_decode = make_values(1, kv_size, 41.0f);
    tiny_llm::Tensor v_decode = make_values(1, kv_size, 51.0f);
    tiny_llm::Tensor out_decode = torch::empty_like(q_decode);
    tiny_llm::Tensor positions_decode = int_cuda({prefill_tokens}, {1});
    tiny_llm::Tensor seq_indices_decode = int_cuda({0}, {1});
    tiny_llm::Tensor context_decode = int_cuda({prefill_tokens + 1}, {1});
    tiny_llm::Tensor blocks_decode = int_cuda(block_table(block_count), {1, 1, block_count});
    tiny_llm::ops::PagedAttentionRuntimeMetadata decode_metadata;
    decode_metadata.seq_indices = &seq_indices_decode;
    decode_metadata.context_lens = &context_decode;
    decode_metadata.block_tables = &blocks_decode;
    decode_metadata.block_size_tokens = block_size_tokens;
    decode_metadata.enabled = true;
    tiny_llm::ops::LlamaAttentionParams decode = prefill;
    decode.positions = &positions_decode;
    decode.q = &q_decode;
    decode.k = &k_decode;
    decode.v = &v_decode;
    decode.out = &out_decode;
    decode.metadata = &decode_metadata;
    tiny_llm::ops::llama_attention_forward(decode);
    check_cuda(cudaDeviceSynchronize(), "decode synchronize");

    unsetenv("TINYLLM_PAGED_ATTENTION_BACKEND");
    return {out_prefill.detach().cpu(), out_decode.detach().cpu()};
}

tiny_llm::Tensor run_two_sequence_prefill_then_decode(bool optimized)
{
    if (optimized)
    {
        setenv("TINYLLM_PAGED_ATTENTION_BACKEND", "cuda", 1);
    }
    else
    {
        setenv("TINYLLM_PAGED_ATTENTION_BACKEND", "torch", 1);
    }

    constexpr int32_t num_attention_heads = 12;
    constexpr int32_t num_key_value_heads = 2;
    constexpr int32_t head_dim = 128;
    constexpr int32_t block_size_tokens = 16;
    constexpr int32_t prefill_tokens_per_seq = 32;
    constexpr int32_t num_seqs = 2;
    constexpr int32_t blocks_per_seq = 3;
    constexpr int32_t total_blocks = num_seqs * blocks_per_seq;
    const int32_t kv_size = num_key_value_heads * head_dim;
    const int32_t hidden_size = num_attention_heads * head_dim;
    const size_t block_floats = 2 * static_cast<size_t>(block_size_tokens) * static_cast<size_t>(kv_size);
    tiny_llm::Tensor pool = torch::zeros({static_cast<int64_t>(total_blocks * block_floats)},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    tiny_llm::BlockAllocator blocks(total_blocks, block_floats * sizeof(float), pool.data_ptr<float>(),
                                    tiny_llm::ParallelConfig::cuda(0));
    tiny_llm::KVCache::Config kv_cfg;
    kv_cfg.num_layers = 1;
    kv_cfg.block_size_tokens = block_size_tokens;
    tiny_llm::KVCache kv_cache(kv_cfg, &blocks);
    tiny_llm::ExecutionContext ctx(nullptr, nullptr, &kv_cache, tiny_llm::ParallelConfig::cuda(0));

    std::vector<int32_t> prefill_positions;
    std::vector<int32_t> prefill_seq_indices;
    for (int32_t seq = 0; seq < num_seqs; ++seq)
    {
        for (int32_t pos = 0; pos < prefill_tokens_per_seq; ++pos)
        {
            prefill_positions.push_back(pos);
            prefill_seq_indices.push_back(seq);
        }
    }
    std::vector<int32_t> block_table_values;
    for (int32_t seq = 0; seq < num_seqs; ++seq)
    {
        for (int32_t block = 0; block < blocks_per_seq; ++block)
        {
            block_table_values.push_back(seq * blocks_per_seq + block);
        }
    }

    const int32_t prefill_rows = prefill_tokens_per_seq * num_seqs;
    tiny_llm::Tensor q_prefill = make_values(prefill_rows, hidden_size, 101.0f);
    tiny_llm::Tensor k_prefill = make_values(prefill_rows, kv_size, 111.0f);
    tiny_llm::Tensor v_prefill = make_values(prefill_rows, kv_size, 121.0f);
    tiny_llm::Tensor out_prefill = torch::empty_like(q_prefill);
    tiny_llm::Tensor positions_prefill = int_cuda(prefill_positions, {prefill_rows});
    tiny_llm::Tensor seq_indices_prefill = int_cuda(prefill_seq_indices, {prefill_rows});
    tiny_llm::Tensor context_prefill = int_cuda({prefill_tokens_per_seq, prefill_tokens_per_seq}, {num_seqs});
    tiny_llm::Tensor blocks_prefill = int_cuda(block_table_values, {1, num_seqs, blocks_per_seq});
    std::vector<tiny_llm::ops::PagedAttentionPrefillSegment> prefill_segments;
    if (optimized)
    {
        prefill_segments.push_back({0, 0, prefill_tokens_per_seq});
        prefill_segments.push_back({prefill_tokens_per_seq, 1, prefill_tokens_per_seq});
    }
    tiny_llm::ops::PagedAttentionRuntimeMetadata prefill_metadata;
    prefill_metadata.seq_indices = &seq_indices_prefill;
    prefill_metadata.context_lens = &context_prefill;
    prefill_metadata.block_tables = &blocks_prefill;
    prefill_metadata.prefill_segments = prefill_segments.empty() ? nullptr : prefill_segments.data();
    prefill_metadata.prefill_segment_count = static_cast<int64_t>(prefill_segments.size());
    prefill_metadata.block_size_tokens = block_size_tokens;
    prefill_metadata.prefill_segments_valid = !prefill_segments.empty();
    prefill_metadata.enabled = true;
    tiny_llm::ops::LlamaAttentionParams prefill;
    prefill.positions = &positions_prefill;
    prefill.q = &q_prefill;
    prefill.k = &k_prefill;
    prefill.v = &v_prefill;
    prefill.out = &out_prefill;
    prefill.ctx = &ctx;
    prefill.metadata = &prefill_metadata;
    prefill.layer_id = 0;
    prefill.num_attention_heads = num_attention_heads;
    prefill.num_key_value_heads = num_key_value_heads;
    prefill.head_dim = head_dim;
    tiny_llm::ops::llama_attention_forward(prefill);
    check_cuda(cudaDeviceSynchronize(), "two sequence prefill synchronize");

    tiny_llm::Tensor q_decode = make_values(num_seqs, hidden_size, 131.0f);
    tiny_llm::Tensor k_decode = make_values(num_seqs, kv_size, 141.0f);
    tiny_llm::Tensor v_decode = make_values(num_seqs, kv_size, 151.0f);
    tiny_llm::Tensor out_decode = torch::empty_like(q_decode);
    tiny_llm::Tensor positions_decode = int_cuda({prefill_tokens_per_seq, prefill_tokens_per_seq}, {num_seqs});
    tiny_llm::Tensor seq_indices_decode = int_cuda({0, 1}, {num_seqs});
    tiny_llm::Tensor context_decode = int_cuda({prefill_tokens_per_seq + 1, prefill_tokens_per_seq + 1}, {num_seqs});
    tiny_llm::Tensor blocks_decode = int_cuda(block_table_values, {1, num_seqs, blocks_per_seq});
    tiny_llm::ops::PagedAttentionRuntimeMetadata decode_metadata;
    decode_metadata.seq_indices = &seq_indices_decode;
    decode_metadata.context_lens = &context_decode;
    decode_metadata.block_tables = &blocks_decode;
    decode_metadata.block_size_tokens = block_size_tokens;
    decode_metadata.enabled = true;
    tiny_llm::ops::LlamaAttentionParams decode = prefill;
    decode.positions = &positions_decode;
    decode.q = &q_decode;
    decode.k = &k_decode;
    decode.v = &v_decode;
    decode.out = &out_decode;
    decode.metadata = &decode_metadata;
    tiny_llm::ops::llama_attention_forward(decode);
    check_cuda(cudaDeviceSynchronize(), "two sequence decode synchronize");

    unsetenv("TINYLLM_PAGED_ATTENTION_BACKEND");
    return out_decode.detach().cpu();
}

void expect_optimized_matches_reference(int32_t num_attention_heads, int32_t num_key_value_heads, int32_t head_dim,
                                        int32_t block_size_tokens, int32_t prefill_tokens, float tolerance)
{
    const CaseResult reference = run_prefill_then_decode(false, num_attention_heads, num_key_value_heads, head_dim,
                                                         block_size_tokens, prefill_tokens);
    const CaseResult optimized = run_prefill_then_decode(true, num_attention_heads, num_key_value_heads, head_dim,
                                                         block_size_tokens, prefill_tokens);
    expect_close(optimized.prefill, reference.prefill, tolerance);
    expect_close(optimized.decode, reference.decode, tolerance);
}
} // namespace

TEST(PagedAttentionCudaTest, OptimizedBackendMatchesReferenceForPrefillAndDecode)
{
    check_cuda(cudaSetDevice(0), "set CUDA device");
    if (!torch::cuda::is_available())
    {
        GTEST_SKIP() << "CUDA is not available.";
    }

    expect_optimized_matches_reference(1, 1, 4, 2, 2, 1e-4f);
    expect_optimized_matches_reference(9, 3, 64, 2, 2, 1e-4f);
}

TEST(PagedAttentionCudaTest, OptimizedBackendMatchesReferenceForBatchedMultiBlockDecode)
{
    check_cuda(cudaSetDevice(0), "set CUDA device");
    if (!torch::cuda::is_available())
    {
        GTEST_SKIP() << "CUDA is not available.";
    }

    const tiny_llm::Tensor reference = run_two_sequence_prefill_then_decode(false);
    const tiny_llm::Tensor optimized = run_two_sequence_prefill_then_decode(true);
    expect_close(optimized, reference, 2e-4f);
}

TEST(PagedAttentionCudaTest, OptimizedBackendMatchesReferenceForQwenShapeLongPrefill)
{
    check_cuda(cudaSetDevice(0), "set CUDA device");
    if (!torch::cuda::is_available())
    {
        GTEST_SKIP() << "CUDA is not available.";
    }

    for (int32_t prefill_tokens : {16, 24, 32, 64, 128})
    {
        SCOPED_TRACE(testing::Message() << "prefill_tokens=" << prefill_tokens);
        expect_optimized_matches_reference(12, 2, 128, 16, prefill_tokens, 2e-4f);
    }
}

TEST(PagedAttentionCudaTest, OptimizedBackendMatchesReferenceAboveSingleBlockThreadLimit)
{
    check_cuda(cudaSetDevice(0), "set CUDA device");
    if (!torch::cuda::is_available())
    {
        GTEST_SKIP() << "CUDA is not available.";
    }

    expect_optimized_matches_reference(1, 1, 8, 16, 1032, 2e-4f);
}

TEST(PagedAttentionCudaTest, BFloat16KvCacheMatchesFloat32Reference)
{
    check_cuda(cudaSetDevice(0), "set CUDA device");
    if (!torch::cuda::is_available())
    {
        GTEST_SKIP() << "CUDA is not available.";
    }

    const CaseResult fp32 = run_prefill_then_decode(false, 12, 2, 128, 16, 24);
    const CaseResult bf16 = run_prefill_then_decode(false, 12, 2, 128, 16, 24, tiny_llm::RuntimeDType::kBFloat16);
    const CaseResult bf16_optimized =
        run_prefill_then_decode(true, 12, 2, 128, 16, 24, tiny_llm::RuntimeDType::kBFloat16);
    expect_close(bf16.prefill, fp32.prefill, 2e-2f);
    expect_close(bf16.decode, fp32.decode, 2e-2f);
    expect_close(bf16_optimized.prefill, fp32.prefill, 2e-2f);
    expect_close(bf16_optimized.decode, fp32.decode, 2e-2f);
}
