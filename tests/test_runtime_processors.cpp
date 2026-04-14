#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>

#include "tiny_llm/core/allocator.h"
#include "tiny_llm/runtime/engine.h"
#include "tiny_llm/runtime/tokenizer.h"

#ifndef TINYLLM_SOURCE_DIR
#error "TINYLLM_SOURCE_DIR must be defined by CMake for test assets."
#endif

int main()
{
    const std::string root = TINYLLM_SOURCE_DIR;
    const std::string vocab_path = root + "/assets/tiny_lm/vocab.txt";
    const std::string ckpt_path = root + "/assets/tiny_lm/tiny_lm_checkpoint.txt";

    tiny_llm::WordPieceTokenizer tokenizer = tiny_llm::WordPieceTokenizer::from_vocab_file(vocab_path);

    tiny_llm::StackAllocator allocator(4 * 1024 * 1024);

    constexpr int32_t kNumLayers = 1;
    constexpr int32_t kBlockSizeTokens = 16;
    constexpr size_t kNumBlocks = 64;
    constexpr size_t kBlockBytes = 256;

    void* kv_pool = std::malloc(kNumBlocks * kBlockBytes);
    assert(kv_pool != nullptr);

    tiny_llm::EngineArgs engine_args;
    engine_args.tokenizer = &tokenizer;
    engine_args.model_type = tiny_llm::EngineModelType::kTinyEmbeddingLM;
    engine_args.tiny_lm_checkpoint_path = ckpt_path;
    engine_args.execution_stream = nullptr;
    engine_args.workspace = &allocator;
    engine_args.kv_num_layers = kNumLayers;
    engine_args.kv_block_size_tokens = kBlockSizeTokens;
    engine_args.kv_num_blocks = kNumBlocks;
    engine_args.kv_block_size_bytes = kBlockBytes;
    engine_args.kv_memory_pool = kv_pool;
    engine_args.max_generated_tokens = 8;
    tiny_llm::LLMEngine engine(engine_args);

    {
        tiny_llm::UserSamplingParams sampling_params;
        sampling_params.max_tokens = 8;
        sampling_params.stop_token_ids = {10};

        const uint64_t req_id = engine.add_request("hello", sampling_params, "external-42");

        tiny_llm::UserOutput last;
        bool seen = false;
        while (engine.has_unfinished_requests())
        {
            const std::vector<tiny_llm::UserOutput> outputs = engine.step();
            for (const tiny_llm::UserOutput& output : outputs)
            {
                if (output.internal_id == req_id)
                {
                    last = output;
                    seen = true;
                }
            }
        }

        assert(seen);
        assert(last.external_id == "external-42");
        assert(last.text == "hello world!");
        assert(last.is_finished);
        assert(last.finish_reason == "stop_token");
    }

    {
        tiny_llm::UserSamplingParams sampling_params;
        sampling_params.max_tokens = 1;

        const uint64_t req_id = engine.add_request("tiny", sampling_params, "external-43");

        tiny_llm::UserOutput last;
        bool seen = false;
        while (engine.has_unfinished_requests())
        {
            const std::vector<tiny_llm::UserOutput> outputs = engine.step();
            for (const tiny_llm::UserOutput& output : outputs)
            {
                if (output.internal_id == req_id)
                {
                    last = output;
                    seen = true;
                }
            }
        }

        assert(seen);
        assert(last.external_id == "external-43");
        assert(last.is_finished);
        assert(last.finish_reason == "length");
        assert(last.generated_token_ids.size() == 1);
    }

    std::free(kv_pool);
    return 0;
}
