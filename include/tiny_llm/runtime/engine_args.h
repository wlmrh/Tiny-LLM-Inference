#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "tiny_llm/models/mini_llama.h"
#include "tiny_llm/runtime/parallel_config.h"
#include "tiny_llm/runtime/scheduler.h"
#include "utils/cuda_compat.h"

namespace tiny_llm {

class ExecutionContext;
class KVCache;
class Model;
class StackAllocator;
class Tokenizer;

enum class EngineModelType {
    kPrebuilt = 0,
    kTinyEmbeddingLM = 1,
    kMiniLLaMA = 2,
    kHFLlamaSafeTensor = 3,
};

/**
 * @brief Aggregates runtime construction handles to reduce repeated parameter passing.
 */
struct EngineArgs {
    // Optional prebuilt handles (legacy/compat path).
    Model* model = nullptr;
    ExecutionContext* ctx = nullptr;
    KVCache* kv = nullptr;
    Tokenizer* tokenizer = nullptr;
    ParallelConfig parallel_config = ParallelConfig::cpu();

    // Model construction inputs (used when model == nullptr).
    EngineModelType model_type = EngineModelType::kPrebuilt;
    std::string tiny_lm_checkpoint_path;
    MiniLLaMAConfig mini_llama_config = MiniLLaMAConfig{};
    std::string hf_model_dir;
    std::string hf_weight_file = "model.safetensors";
    int32_t max_batch_size = 1;

    // ExecutionContext construction inputs (used when ctx == nullptr).
    cudaStream_t execution_stream = nullptr;
    StackAllocator* workspace = nullptr;
    size_t workspace_pool_size = 0;

    // KV recursive construction inputs.
    int32_t kv_num_layers = 0;
    int32_t kv_block_size_tokens = 16;
    size_t kv_num_blocks = 0;
    size_t kv_block_size_bytes = 0;
    void* kv_memory_pool = nullptr;

    int32_t max_generated_tokens = 32;
    SchedulerConfig scheduler_config = SchedulerConfig{};
};

} // namespace tiny_llm
