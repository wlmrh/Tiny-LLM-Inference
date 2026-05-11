#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include "tiny_llm/core/allocator.h"
#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/runtime/engine.h"
#include "tiny_llm/runtime/tokenizer.h"

#ifndef TINYLLM_SOURCE_DIR
#error "TINYLLM_SOURCE_DIR must be defined by CMake for test assets."
#endif

std::filesystem::path expand_user_path(std::string path_str) {
    if (path_str.empty()) return {};

    // 检查是否以 ~ 开头
    if (path_str[0] == '~') {
        const char* home = std::getenv("HOME"); // Linux/macOS
        if (!home) {
            // 如果是在 Windows 环境下调试
            const char* drive = std::getenv("HOMEDRIVE");
            const char* home_path = std::getenv("HOMEPATH");
            if (drive && home_path) {
                return std::filesystem::path(std::string(drive) + std::string(home_path)) / path_str.substr(2);
            }
            return path_str; // 实在找不到，原样返回
        }
        
        // 拼接路径：注意处理 ~/models 和 ~models 两种情况
        if (path_str.size() == 1) return std::filesystem::path(home);
        return std::filesystem::path(home) / path_str.substr(2); // 跳过 "~/"
    }
    
    return std::filesystem::path(path_str);
}

size_t llama_kv_block_bytes(const tiny_llm::LlamaConfig& config, int32_t block_size_tokens)
{
    if (block_size_tokens <= 0 || config.head_dim <= 0 || config.num_key_value_heads <= 0)
    {
        throw std::runtime_error("invalid KV cache dimensions.");
    }
    const size_t kv_hidden_size =
        static_cast<size_t>(config.num_key_value_heads) * static_cast<size_t>(config.head_dim);
    return 2 * static_cast<size_t>(block_size_tokens) * kv_hidden_size * sizeof(float);
}

int main(int argc, char** argv)
{
    std::string hf_model_dir;

    if (argc > 1) {
        hf_model_dir = argv[1];
    } 
    else if (const char* env_hf_dir = std::getenv("TINYLLM_HF_TINY_LLAMA_DIR")) {
        hf_model_dir = env_hf_dir;
    } 
    else {
        std::cerr << "====================================================\n";
        std::cerr << "❌ 错误: 未指定模型路径!\n";
        std::cerr << "用法: " << argv[0] << " <hf_model_dir_path>\n";
        std::cerr << "或者设置环境变量: export TINYLLM_HF_TINY_LLAMA_DIR=...\n";
        std::cerr << "====================================================\n";
        return 77;
    }

    hf_model_dir = expand_user_path(hf_model_dir);

    if (!std::filesystem::exists(hf_model_dir)) {
        std::cerr << "❌ 错误: 模型文件夹不存在 -> " << hf_model_dir << "\n";
        return 77;
    }
    
    if (!std::filesystem::exists(hf_model_dir + "/config.json")) {
        std::cerr << "❌ 错误: 找不到 config.json\n";
        return 1;
    }
    if (!std::filesystem::exists(hf_model_dir + "/model.safetensors")) {
        std::cerr << "❌ 错误: 找不到 model.safetensors\n";
        return 1;
    }
    if (!std::filesystem::exists(hf_model_dir + "/tokenizer.json")) {
        std::cerr << "❌ 错误: 找不到 tokenizer.json\n";
        return 1;
    }
    if (!std::filesystem::exists(hf_model_dir + "/tokenizer.json")
        && !std::filesystem::exists(hf_model_dir + "/tokenizer.model")) {
        std::cerr << "❌ 错误: 找不到 tokenizer.json 或 tokenizer.model\n";
        return 1;
    }

    std::cout << "✅ 成功加载模型目录: " << hf_model_dir << "\n";

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

#if TINYLLM_ENABLE_CUDA
    const tiny_llm::ParallelConfig runtime_parallel_config = tiny_llm::ParallelConfig::cuda(0);
#else
    const tiny_llm::ParallelConfig runtime_parallel_config = tiny_llm::ParallelConfig::cpu();
#endif

    tiny_llm::StackAllocator allocator(4 * 1024 * 1024, runtime_parallel_config);

    constexpr int32_t kBlockSizeTokens = 16;
    constexpr size_t kNumBlocks = 64;
    const size_t kBlockBytes = llama_kv_block_bytes(hf_config, kBlockSizeTokens);

    void* kv_pool = nullptr;
#if TINYLLM_ENABLE_CUDA
    if (runtime_parallel_config.is_cuda()) {
        cudaError_t status = cudaSetDevice(runtime_parallel_config.device_id());
        assert(status == cudaSuccess);
        status = cudaMalloc(&kv_pool, kNumBlocks * kBlockBytes);
        assert(status == cudaSuccess);
    } else
#endif
    {
        kv_pool = std::malloc(kNumBlocks * kBlockBytes);
    }
    assert(kv_pool != nullptr);

    tiny_llm::EngineArgs engine_args;
    engine_args.tokenizer = &tokenizer;
    engine_args.parallel_config = runtime_parallel_config;
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

#if TINYLLM_ENABLE_CUDA
    if (runtime_parallel_config.is_cuda()) {
        cudaError_t status = cudaFree(kv_pool);
        assert(status == cudaSuccess);
    } else
#endif
    {
        std::free(kv_pool);
    }
    return 0;
}
