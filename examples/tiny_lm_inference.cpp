#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "tiny_llm/core/allocator.h"
#include "tiny_llm/core/context.h"
#include "tiny_llm/models/tiny_lm.h"
#include "tiny_llm/runtime/engine.h"
#include "tiny_llm/runtime/tokenizer.h"

#if TINYLLM_ENABLE_CUDA
#include "utils/cuda_utils.h"
#endif

int main(int argc, char** argv)
{
    const std::string vocab_path = (argc > 1) ? argv[1] : "assets/tiny_lm/vocab.txt";
    const std::string checkpoint_path = (argc > 2) ? argv[2] : "assets/tiny_lm/tiny_lm_checkpoint.txt";
    const std::string prompt = (argc > 3) ? argv[3] : "hello";

    try
    {
        tiny_llm::WordPieceTokenizer tokenizer = tiny_llm::WordPieceTokenizer::from_vocab_file(vocab_path);
        tiny_llm::TinyEmbeddingLM model = tiny_llm::TinyEmbeddingLM::from_checkpoint(checkpoint_path);

        tiny_llm::StackAllocator allocator(16 * 1024 * 1024);

        constexpr int32_t kBlockSizeTokens = 16;
        constexpr size_t kNumBlocks = 128;
        constexpr size_t kBlockBytes = 256;

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
        tiny_llm::EngineArgs engine_args;
        engine_args.model = &model;
        engine_args.ctx = &ctx;
        engine_args.tokenizer = &tokenizer;
        engine_args.kv_num_layers = model.num_layers();
        engine_args.kv_block_size_tokens = kBlockSizeTokens;
        engine_args.kv_num_blocks = kNumBlocks;
        engine_args.kv_block_size_bytes = kBlockBytes;
        engine_args.kv_memory_pool = kv_pool;
        engine_args.max_generated_tokens = 12;
        tiny_llm::LLMEngine engine(engine_args);

        tiny_llm::UserSamplingParams sampling_params;
        sampling_params.max_tokens = 12;
        const uint64_t req_id = engine.add_request(prompt, sampling_params, "demo-1");

        std::string final_text;
        while (engine.has_unfinished_requests())
        {
                const std::vector<tiny_llm::UserOutput> outputs = engine.step();
                for (const tiny_llm::UserOutput& output : outputs)
                {
                        if (output.internal_id == req_id)
                        {
                                final_text = output.text;
                        }
                }
        }

        std::cout << "Tiny LM runtime demo" << std::endl;
        std::cout << "prompt: " << prompt << std::endl;
        std::cout << "output: " << final_text << std::endl;

#if TINYLLM_ENABLE_CUDA
        CHECK_CUDA(cudaStreamDestroy(stream));
        CHECK_CUDA(cudaFree(kv_pool));
#else
        std::free(kv_pool);
#endif
    }
    catch (const std::exception& e)
    {
        std::cerr << "tiny_lm_inference failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
