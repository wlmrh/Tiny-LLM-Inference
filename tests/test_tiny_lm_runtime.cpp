#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "tiny_llm/core/allocator.h"
#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/runtime/engine.h"
#include "tiny_llm/runtime/tokenizer.h"

#ifndef TINYLLM_SOURCE_DIR
#error "TINYLLM_SOURCE_DIR must be defined by CMake for test assets."
#endif

#ifndef TINYLLM_HF_TINY_LLAMA_DIR
#error "TINYLLM_HF_TINY_LLAMA_DIR must be defined by CMake for HF tiny llama model path."
#endif

int main()
{
    std::string hf_model_dir;
    if (const char* env_hf_dir = std::getenv("TINYLLM_HF_TINY_LLAMA_DIR"))
    {
        hf_model_dir = env_hf_dir;
    }
    if (hf_model_dir.empty())
    {
        hf_model_dir = TINYLLM_HF_TINY_LLAMA_DIR;
    }
    if (!std::filesystem::exists(hf_model_dir))
    {
        constexpr const char* kFallbackDir = "/Users/tangqi/weights";
        hf_model_dir = kFallbackDir;
    }

    assert(std::filesystem::exists(hf_model_dir));
    assert(std::filesystem::exists(hf_model_dir + "/config.json"));
    assert(std::filesystem::exists(hf_model_dir + "/model.safetensors"));
    const bool has_tokenizer_json = std::filesystem::exists(hf_model_dir + "/tokenizer.json");
    const bool has_tokenizer_model = std::filesystem::exists(hf_model_dir + "/tokenizer.model");
    assert(has_tokenizer_json || has_tokenizer_model);

    const tiny_llm::LlamaConfig hf_config =
        tiny_llm::HFLlamaConfigLoader::load_from_dir(hf_model_dir);
    tiny_llm::HFLlamaTokenizer tokenizer = tiny_llm::HFLlamaTokenizer::from_model_dir(hf_model_dir);
    assert(tokenizer.vocab_size() == hf_config.vocab_size);
    assert(tokenizer.is_valid_token_id(tokenizer.bos_id()));
    assert(tokenizer.is_valid_token_id(tokenizer.eos_id()));
    assert(tokenizer.is_valid_token_id(tokenizer.unk_id()));
    const std::vector<int32_t> hf_ids = tokenizer.encode("hello tiny inference");
    assert(!hf_ids.empty());
    const std::string roundtrip = tokenizer.decode(hf_ids);
    assert(!roundtrip.empty());
    assert(roundtrip.find("hello") != std::string::npos);

    tiny_llm::StackAllocator allocator(4 * 1024 * 1024);

    constexpr int32_t kBlockSizeTokens = 16;
    constexpr size_t kNumBlocks = 64;
    constexpr size_t kBlockBytes = 256;

    void* kv_pool = std::malloc(kNumBlocks * kBlockBytes);
    assert(kv_pool != nullptr);

    tiny_llm::EngineArgs engine_args;
    engine_args.tokenizer = &tokenizer;
    engine_args.model_type = tiny_llm::EngineModelType::kHFLlamaSafeTensor;
    engine_args.hf_model_dir = hf_model_dir;
    engine_args.hf_weight_file = "model.safetensors";
    engine_args.execution_stream = nullptr;
    engine_args.workspace = &allocator;
    engine_args.max_batch_size = 8;
    engine_args.kv_num_layers = hf_config.num_hidden_layers;
    engine_args.kv_block_size_tokens = kBlockSizeTokens;
    engine_args.kv_num_blocks = kNumBlocks;
    engine_args.kv_block_size_bytes = kBlockBytes;
    engine_args.kv_memory_pool = kv_pool;
    engine_args.max_generated_tokens = 8;
    tiny_llm::LLMEngine engine(engine_args);

    tiny_llm::UserSamplingParams sampling_params;
    sampling_params.max_tokens = 8;

    const std::string prompt_1 = "hello";
    const uint64_t req_1 = engine.add_request(prompt_1, sampling_params, "req-1");
    std::string out_1;
    int step_count_1 = 0;
    while (engine.has_unfinished_requests())
    {
        ++step_count_1;
        const std::vector<tiny_llm::UserOutput> outputs = engine.step();
        for (const tiny_llm::UserOutput& output : outputs)
        {
            if (output.internal_id == req_1)
            {
                out_1 = output.text;
            }
        }
    }
    assert(step_count_1 > 0);
    assert(!out_1.empty());
    std::cout << "[test_tiny_lm_runtime] prompt: " << prompt_1 << "\n";
    std::cout << "[test_tiny_lm_runtime] output: " << out_1 << "\n";

    const std::string prompt_2 = "tiny llm inference";
    const uint64_t req_2 = engine.add_request(prompt_2, sampling_params, "req-2");
    std::string out_2;
    int step_count_2 = 0;
    while (engine.has_unfinished_requests())
    {
        ++step_count_2;
        const std::vector<tiny_llm::UserOutput> outputs = engine.step();
        for (const tiny_llm::UserOutput& output : outputs)
        {
            if (output.internal_id == req_2)
            {
                out_2 = output.text;
            }
        }
    }
    assert(step_count_2 > 0);
    assert(!out_2.empty());
    std::cout << "[test_tiny_lm_runtime] prompt: " << prompt_2 << "\n";
    std::cout << "[test_tiny_lm_runtime] output: " << out_2 << "\n";

    std::free(kv_pool);
    return 0;
}
