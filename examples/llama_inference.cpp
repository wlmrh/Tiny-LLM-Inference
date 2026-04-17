#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/allocator.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/models/mini_llama.h"
#include "tiny_llm/runtime/engine.h"
#include "tiny_llm/runtime/tokenizer.h"

#if TINYLLM_ENABLE_CUDA
#include "utils/cuda_utils.h"
#endif

int main() {
    tiny_llm::StackAllocator allocator(64 * 1024 * 1024);

    constexpr int32_t kNumLayers = 4;
    constexpr int32_t kBlockSizeTokens = 16;
    constexpr size_t kNumBlocks = 1024;
    constexpr size_t kBlockBytes = 1024;

    void* kv_pool = nullptr;
#if TINYLLM_ENABLE_CUDA
    CHECK_CUDA(cudaMalloc(&kv_pool, kNumBlocks * kBlockBytes));
#else
    kv_pool = std::malloc(kNumBlocks * kBlockBytes);
    if (kv_pool == nullptr)
    {
        std::cerr << "failed to allocate KV pool" << std::endl;
        return 1;
    }
#endif

#if TINYLLM_ENABLE_CUDA
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
#else
    cudaStream_t stream = nullptr;
#endif

    tiny_llm::ExecutionContext ctx(stream, &allocator, nullptr);
    tiny_llm::MiniLLaMA model(tiny_llm::MiniLLaMAConfig{});
    tiny_llm::WhitespaceTokenizer tokenizer;
    tiny_llm::EngineArgs engine_args;
    engine_args.model = &model;
    engine_args.ctx = &ctx;
    engine_args.tokenizer = &tokenizer;
    engine_args.kv_num_layers = kNumLayers;
    engine_args.kv_block_size_tokens = kBlockSizeTokens;
    engine_args.kv_num_blocks = kNumBlocks;
    engine_args.kv_block_size_bytes = kBlockBytes;
    engine_args.kv_memory_pool = kv_pool;
    tiny_llm::LLMEngine engine(engine_args);

    tiny_llm::UserSamplingParams sampling_params;
    sampling_params.max_tokens = 8;
    const uint64_t req_1 = engine.add_request("tiny llm on single gpu", sampling_params, "req-1");
    const uint64_t req_2 = engine.add_request("paged kv cache demo", sampling_params, "req-2");

    std::string text_1;
    std::string text_2;
    while (engine.has_unfinished_requests())
    {
        const std::vector<tiny_llm::UserOutput> outputs = engine.step();
        for (const tiny_llm::UserOutput& output : outputs)
        {
            if (output.internal_id == req_1)
            {
                text_1 = output.text;
            }
            else if (output.internal_id == req_2)
            {
                text_2 = output.text;
            }
        }
    }

    std::cout << "Tiny-LLM inference example runtime demo" << std::endl;
    std::cout << "seq 1: " << text_1 << std::endl;
    std::cout << "seq 2: " << text_2 << std::endl;

#if TINYLLM_ENABLE_CUDA
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(kv_pool));
#else
    std::free(kv_pool);
#endif

    return 0;
}
