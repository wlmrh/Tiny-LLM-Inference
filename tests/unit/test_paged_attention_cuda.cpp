#include "tiny_llm/core/allocator.h"
#include "tiny_llm/core/context.h"
#include "tiny_llm/operators/paged_attention.h"
#include "tiny_llm/runtime/kv_cache.h"
#include "tiny_llm/runtime/parallel_config.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* message)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

void expect_close(const tiny_llm::Tensor& actual, const tiny_llm::Tensor& expected, float tolerance, const char* label)
{
    tiny_llm::Tensor a = actual.cpu().contiguous();
    tiny_llm::Tensor e = expected.cpu().contiguous();
    if (a.numel() != e.numel())
    {
        throw std::runtime_error(std::string(label) + ": numel mismatch.");
    }
    const float* ap = a.data_ptr<float>();
    const float* ep = e.data_ptr<float>();
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        if (std::fabs(ap[i] - ep[i]) > tolerance)
        {
            throw std::runtime_error(std::string(label) + ": value mismatch.");
        }
    }
}

tiny_llm::Tensor int_cuda(std::vector<int32_t> values, std::vector<int64_t> shape)
{
    return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32)).reshape(shape).to(torch::kCUDA);
}

tiny_llm::Tensor make_values(int64_t rows, int64_t cols, float offset)
{
    tiny_llm::Tensor values = torch::arange(
        rows * cols,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    return (values.reshape({rows, cols}) + offset) / 100.0f;
}

struct CaseResult {
    tiny_llm::Tensor prefill;
    tiny_llm::Tensor decode;
};

CaseResult run_prefill_then_decode(bool optimized,
                                   int32_t num_attention_heads,
                                   int32_t num_key_value_heads,
                                   int32_t head_dim)
{
    if (optimized)
    {
        setenv("TINYLLM_PAGED_ATTENTION_BACKEND", "cuda", 1);
    }
    else
    {
        unsetenv("TINYLLM_PAGED_ATTENTION_BACKEND");
    }

    constexpr int32_t kBlockSizeTokens = 2;
    constexpr int32_t kNumBlocks = 2;
    const int32_t kv_size = num_key_value_heads * head_dim;
    const int32_t hidden_size = num_attention_heads * head_dim;
    const size_t block_floats = 2 * static_cast<size_t>(kBlockSizeTokens) * static_cast<size_t>(kv_size);
    tiny_llm::Tensor pool = torch::zeros(
        {static_cast<int64_t>(kNumBlocks * block_floats)},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    tiny_llm::BlockAllocator blocks(
        kNumBlocks,
        block_floats * sizeof(float),
        pool.data_ptr<float>(),
        tiny_llm::ParallelConfig::cuda(0));
    tiny_llm::KVCache::Config kv_cfg;
    kv_cfg.num_layers = 1;
    kv_cfg.block_size_tokens = kBlockSizeTokens;
    tiny_llm::KVCache kv_cache(kv_cfg, &blocks);
    tiny_llm::ExecutionContext ctx(nullptr, nullptr, &kv_cache, tiny_llm::ParallelConfig::cuda(0));

    tiny_llm::Tensor q_prefill = make_values(2, hidden_size, 1.0f);
    tiny_llm::Tensor k_prefill = make_values(2, kv_size, 11.0f);
    tiny_llm::Tensor v_prefill = make_values(2, kv_size, 21.0f);
    tiny_llm::Tensor out_prefill = torch::empty_like(q_prefill);
    tiny_llm::Tensor positions_prefill = int_cuda({0, 1}, {2});
    tiny_llm::Tensor seq_indices_prefill = int_cuda({0, 0}, {2});
    tiny_llm::Tensor context_prefill = int_cuda({2}, {1});
    tiny_llm::Tensor blocks_prefill = int_cuda({0, -1}, {1, 1, 2});
    tiny_llm::ops::PagedAttentionRuntimeMetadata prefill_metadata;
    prefill_metadata.seq_indices = &seq_indices_prefill;
    prefill_metadata.context_lens = &context_prefill;
    prefill_metadata.block_tables = &blocks_prefill;
    prefill_metadata.block_size_tokens = kBlockSizeTokens;
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
    tiny_llm::Tensor positions_decode = int_cuda({2}, {1});
    tiny_llm::Tensor seq_indices_decode = int_cuda({0}, {1});
    tiny_llm::Tensor context_decode = int_cuda({3}, {1});
    tiny_llm::Tensor blocks_decode = int_cuda({0, 1}, {1, 1, 2});
    tiny_llm::ops::PagedAttentionRuntimeMetadata decode_metadata;
    decode_metadata.seq_indices = &seq_indices_decode;
    decode_metadata.context_lens = &context_decode;
    decode_metadata.block_tables = &blocks_decode;
    decode_metadata.block_size_tokens = kBlockSizeTokens;
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

} // namespace

int main()
{
    check_cuda(cudaSetDevice(0), "set CUDA device");
    if (!torch::cuda::is_available())
    {
        return 77;
    }

    const CaseResult reference_small = run_prefill_then_decode(false, 1, 1, 4);
    const CaseResult optimized_small = run_prefill_then_decode(true, 1, 1, 4);
    expect_close(optimized_small.prefill, reference_small.prefill, 1e-4f, "small prefill parity");
    expect_close(optimized_small.decode, reference_small.decode, 1e-4f, "small decode parity");

    const CaseResult reference_smol = run_prefill_then_decode(false, 9, 3, 64);
    const CaseResult optimized_smol = run_prefill_then_decode(true, 9, 3, 64);
    expect_close(optimized_smol.prefill, reference_smol.prefill, 1e-4f, "smollm prefill parity");
    expect_close(optimized_smol.decode, reference_smol.decode, 1e-4f, "smollm decode parity");

    return 0;
}
