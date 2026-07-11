#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tiny_llm/runtime/parallel_config.h"
#include "tiny_llm/runtime/processors.h"
#include "tiny_llm/runtime/runtime_stats.h"
#include "tiny_llm/runtime/runtime_dtype.h"
#include "tiny_llm/runtime/scheduler.h"

namespace tiny_llm
{

class HFLlamaTokenizer;
class LLMEngine;
class StackAllocator;

/**
 * @brief Construction options for the offline C++ LLM convenience wrapper.
 *
 * This layer owns the tokenizer, workspace, KV block pool, and LLMEngine. It is
 * intended for deployment-style usage where callers should not manually wire the
 * runtime pieces used by integration tests.
 */
struct LLMOptions
{
    LLMOptions() = default;
    explicit LLMOptions(std::string model_path);
    LLMOptions(std::string model_path, ParallelConfig parallel_config);

    std::string model;
    ParallelConfig parallel_config = ParallelConfig::cpu();
    RuntimeDType compute_dtype = RuntimeDType::kFloat32;
    RuntimeDType kv_cache_dtype = RuntimeDType::kFloat32;
    std::string weight_file = "model.safetensors";

    int32_t max_num_seqs = 16;
    int32_t max_tokens = 32;

    int32_t block_size_tokens = 16;
    size_t kv_num_blocks = 256;
    size_t workspace_pool_size = 16 * 1024 * 1024;

    SchedulerConfig scheduler_config = SchedulerConfig{};
};

using LLMSamplingParams = UserSamplingParams;

struct CompletionOutput
{
    std::string prompt;
    std::string text;
    std::vector<int32_t> token_ids;
    bool finished = false;
    std::string finish_reason;
};

struct CompletionStreamOutput : public CompletionOutput
{
    size_t prompt_index = 0;
    std::string delta_text;
    int32_t token_id = -1;
};

using CompletionStreamCallback = std::function<void(const CompletionStreamOutput &)>;

/**
 * @brief vLLM-style offline generation facade for C++ callers.
 */
class LLM
{
  public:
    explicit LLM(std::string model);
    LLM(std::string model, ParallelConfig parallel_config);
    explicit LLM(LLMOptions options);
    ~LLM();

    LLM(LLM &&) noexcept;
    LLM &operator=(LLM &&) noexcept;
    LLM(const LLM &) = delete;
    LLM &operator=(const LLM &) = delete;

    std::vector<CompletionOutput> generate(const std::vector<std::string> &prompts,
                                           const LLMSamplingParams &sampling_params = LLMSamplingParams{},
                                           CompletionStreamCallback callback = CompletionStreamCallback{});
    CompletionOutput generate(const std::string &prompt, const LLMSamplingParams &sampling_params = LLMSamplingParams{},
                              CompletionStreamCallback callback = CompletionStreamCallback{});

    const RuntimeProfilingStats &last_generation_profile() const
    {
        return last_generation_profile_;
    }

  private:
    void initialize();
    void release_kv_pool() noexcept;

    LLMOptions options_;
    std::unique_ptr<HFLlamaTokenizer> tokenizer_;
    std::unique_ptr<StackAllocator> workspace_;
    std::unique_ptr<LLMEngine> engine_;
    void *kv_pool_ = nullptr;
    RuntimeProfilingStats last_generation_profile_;
};

} // namespace tiny_llm
