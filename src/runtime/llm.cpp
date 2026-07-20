#include "tiny_llm/runtime/llm.h"

#include "tiny_llm/core/allocator.h"
#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/runtime/engine.h"
#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/tokenizer.h"

#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace tiny_llm
{
namespace
{

std::string expand_user_path(const std::string &path)
{
    if (path.empty() || path[0] != '~')
    {
        return path;
    }

    const char *home = std::getenv("HOME");
    if (home == nullptr || *home == '\0')
    {
        throw std::runtime_error("LLM: cannot expand '~' because HOME is not set.");
    }

    if (path.size() == 1)
    {
        return std::string(home);
    }
    if (path[1] == '/')
    {
        return std::string(home) + path.substr(1);
    }

    return path;
}

size_t checked_mul(size_t lhs, size_t rhs, const std::string &field)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        throw std::runtime_error("LLM: size overflow while computing " + field + ".");
    }
    return lhs * rhs;
}

size_t llama_kv_block_bytes(const LlamaConfig &config, int32_t block_size_tokens, RuntimeDType kv_cache_dtype)
{
    if (block_size_tokens <= 0)
    {
        throw std::runtime_error("LLM: block_size_tokens must be positive.");
    }
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.hidden_size <= 0)
    {
        throw std::runtime_error("LLM: invalid LLaMA dimensions in model config.");
    }
    if (config.hidden_size % config.num_attention_heads != 0)
    {
        throw std::runtime_error("LLM: hidden_size must be divisible by num_attention_heads.");
    }

    const int32_t head_dim = config.hidden_size / config.num_attention_heads; // dimension of each head
    const size_t kv_hidden_size = // hidden dimension of K/V cache for each token
        checked_mul(static_cast<size_t>(config.num_key_value_heads), static_cast<size_t>(head_dim), "kv_hidden_size");
    const size_t tokens = static_cast<size_t>(block_size_tokens);
    return checked_mul(checked_mul(2, tokens, "kv block tokens"),
                       checked_mul(kv_hidden_size, runtime_dtype_size(kv_cache_dtype), "kv block width"),
                       "kv block bytes");
}

bool has_any_safetensors_file(const std::filesystem::path &model_dir)
{
    if (!std::filesystem::exists(model_dir) || !std::filesystem::is_directory(model_dir))
    {
        return false;
    }
    for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(model_dir))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors")
        {
            return true;
        }
    }
    return false;
}

void validate_model_dir(const std::filesystem::path &model_dir, const std::string &weight_file)
{
    if (!std::filesystem::is_directory(model_dir))
    {
        throw std::runtime_error("LLM: model path is not a directory: " + model_dir.string());
    }
    if (!std::filesystem::exists(model_dir / "config.json"))
    {
        throw std::runtime_error("LLM: missing config.json under model directory: " + model_dir.string());
    }
    const bool uses_default_weight = weight_file.empty() || weight_file == "model.safetensors";
    if (!uses_default_weight && !std::filesystem::exists(model_dir / weight_file))
    {
        throw std::runtime_error("LLM: missing weight file under model directory: " +
                                 (model_dir / weight_file).string());
    }
    if (uses_default_weight && !std::filesystem::exists(model_dir / "model.safetensors") &&
        !has_any_safetensors_file(model_dir))
    {
        throw std::runtime_error("LLM: missing safetensors weight file under model directory: " + model_dir.string());
    }
    if (!std::filesystem::exists(model_dir / "tokenizer.json") &&
        !std::filesystem::exists(model_dir / "tokenizer.model"))
    {
        throw std::runtime_error("LLM: missing tokenizer.json or tokenizer.model under model directory: " +
                                 model_dir.string());
    }
}

void *allocate_kv_pool(size_t bytes, const ParallelConfig &parallel_config)
{
    if (bytes == 0)
    {
        throw std::runtime_error("LLM: KV pool size must be non-zero.");
    }

    if (parallel_config.is_cpu())
    {
        void *ptr = std::malloc(bytes);
        if (ptr == nullptr)
        {
            throw std::runtime_error("LLM: failed to allocate CPU KV block pool.");
        }
        return ptr;
    }

#if TINYLLM_ENABLE_CUDA
    void *ptr = nullptr;
    if (cudaSetDevice(parallel_config.device_id()) != cudaSuccess)
    {
        throw std::runtime_error("LLM: failed to set CUDA device for KV block pool.");
    }
    if (cudaMalloc(&ptr, bytes) != cudaSuccess)
    {
        throw std::runtime_error("LLM: failed to allocate CUDA KV block pool.");
    }
    return ptr;
#else
    throw std::runtime_error("LLM: CUDA device requested, but this build was compiled without CUDA.");
#endif
}

} // namespace

LLMOptions::LLMOptions(std::string model_path) : model(std::move(model_path)) {}

LLMOptions::LLMOptions(std::string model_path, ParallelConfig parallel_config)
    : model(std::move(model_path)), parallel_config(parallel_config)
{
}

LLM::LLM(std::string model) : options_(std::move(model))
{
    initialize();
}

LLM::LLM(std::string model, ParallelConfig parallel_config) : options_(std::move(model), parallel_config)
{
    initialize();
}

LLM::LLM(LLMOptions options) : options_(std::move(options))
{
    initialize();
}

LLM::~LLM()
{
    engine_.reset();
    workspace_.reset();
    tokenizer_.reset();
    release_kv_pool();
}

LLM::LLM(LLM &&other) noexcept
    : options_(std::move(other.options_)), tokenizer_(std::move(other.tokenizer_)),
      workspace_(std::move(other.workspace_)), engine_(std::move(other.engine_)), kv_pool_(other.kv_pool_),
      last_step_profile_(other.last_step_profile_), last_generation_profile_(other.last_generation_profile_)
{
    other.kv_pool_ = nullptr;
    other.last_step_profile_ = RuntimeProfilingStats{};
    other.last_generation_profile_ = RuntimeProfilingStats{};
}

LLM &LLM::operator=(LLM &&other) noexcept
{
    if (this != &other)
    {
        engine_.reset();
        workspace_.reset();
        tokenizer_.reset();
        release_kv_pool();

        options_ = std::move(other.options_);
        tokenizer_ = std::move(other.tokenizer_);
        workspace_ = std::move(other.workspace_);
        engine_ = std::move(other.engine_);
        kv_pool_ = other.kv_pool_;
        last_step_profile_ = other.last_step_profile_;
        last_generation_profile_ = other.last_generation_profile_;
        other.kv_pool_ = nullptr;
        other.last_step_profile_ = RuntimeProfilingStats{};
        other.last_generation_profile_ = RuntimeProfilingStats{};
    }
    return *this;
}

void LLM::initialize()
{
    constexpr const char *kErr = "LLM";
    options_.model = expand_user_path(options_.model);
    if (options_.model.empty())
    {
        throw std::runtime_error(std::string(kErr) + ": model path must be non-empty.");
    }
    if (options_.max_num_seqs <= 0)
    {
        throw std::runtime_error(std::string(kErr) + ": max_num_seqs must be positive.");
    }
    if (options_.max_tokens <= 0)
    {
        throw std::runtime_error(std::string(kErr) + ": max_tokens must be positive.");
    }
    if (options_.kv_num_blocks == 0)
    {
        throw std::runtime_error(std::string(kErr) + ": kv_num_blocks must be positive.");
    }
    if (options_.workspace_pool_size == 0)
    {
        throw std::runtime_error(std::string(kErr) + ": workspace_pool_size must be positive.");
    }

    options_.parallel_config.validate();
    if ((options_.compute_dtype == RuntimeDType::kBFloat16 || options_.kv_cache_dtype == RuntimeDType::kBFloat16) &&
        options_.parallel_config.is_cpu())
    {
        throw std::runtime_error("LLM: bfloat16 compute and KV cache require a CUDA device.");
    }
    const std::filesystem::path model_dir(options_.model);
    validate_model_dir(model_dir, options_.weight_file);

    const LlamaConfig config = HFLlamaConfigLoader::load_from_dir(options_.model);
    // bytes of one physical KV block
    const size_t kv_block_bytes = llama_kv_block_bytes(config, options_.block_size_tokens, options_.kv_cache_dtype);
    // bytes in the complete KV-cache memory pool
    const size_t kv_pool_bytes = checked_mul(options_.kv_num_blocks, kv_block_bytes, "KV pool bytes");

    try
    {
        tokenizer_ = std::make_unique<HFLlamaTokenizer>(HFLlamaTokenizer::from_model_dir(options_.model));
        // space allocated for one inference step
        workspace_ = std::make_unique<StackAllocator>(options_.workspace_pool_size, options_.parallel_config);
        // space allocated for KV cache
        kv_pool_ = allocate_kv_pool(kv_pool_bytes, options_.parallel_config);

        SchedulerConfig scheduler_config = options_.scheduler_config;
        if (scheduler_config.max_running_requests == 0)
        {
            scheduler_config.max_running_requests = options_.max_num_seqs;
        }
        if (scheduler_config.max_prefill_tokens_per_step <= 0)
        {
            throw std::runtime_error(std::string(kErr) +
                                     ": scheduler_config.max_prefill_tokens_per_step must be positive.");
        }

        EngineArgs args;
        args.tokenizer = tokenizer_.get();
        args.parallel_config = options_.parallel_config;
        args.compute_dtype = options_.compute_dtype;
        args.kv_cache_dtype = options_.kv_cache_dtype;
        args.model_type = EngineModelType::kHFLlamaSafeTensor;
        args.hf_model_dir = options_.model;
        args.hf_weight_file = options_.weight_file;
        args.max_batch_size = options_.max_num_seqs;
        args.workspace = workspace_.get();
        args.kv_num_layers = config.num_hidden_layers;
        args.kv_block_size_tokens = options_.block_size_tokens;
        args.kv_num_blocks = options_.kv_num_blocks;
        args.kv_block_size_bytes = kv_block_bytes;
        args.kv_memory_pool = kv_pool_;
        args.max_generated_tokens = options_.max_tokens;
        args.scheduler_config = scheduler_config;

        engine_ = std::make_unique<LLMEngine>(args);
    }
    catch (...)
    {
        engine_.reset();
        workspace_.reset();
        tokenizer_.reset();
        release_kv_pool();
        throw;
    }
}

void LLM::release_kv_pool() noexcept
{
    if (kv_pool_ == nullptr)
    {
        return;
    }

    if (options_.parallel_config.is_cuda())
    {
#if TINYLLM_ENABLE_CUDA
        cudaSetDevice(options_.parallel_config.device_id());
        cudaFree(kv_pool_);
#endif
    }
    else
    {
        std::free(kv_pool_);
    }
    kv_pool_ = nullptr;
}

std::vector<CompletionOutput> LLM::generate(const std::vector<std::string> &prompts,
                                            const LLMSamplingParams &sampling_params, CompletionStreamCallback callback)
{
    if (engine_ == nullptr)
    {
        throw std::runtime_error("LLM::generate: engine is not initialized.");
    }
    if (prompts.empty())
    {
        return {};
    }
    if (has_unfinished_requests())
    {
        throw std::runtime_error("LLM::generate: cannot start batch generation while incremental requests are active.");
    }

    last_generation_profile_ = RuntimeProfilingStats{};

    std::vector<CompletionOutput> results(prompts.size());
    std::unordered_map<uint64_t, size_t> request_to_index;
    request_to_index.reserve(prompts.size());

    for (size_t i = 0; i < prompts.size(); ++i)
    {
        const uint64_t request_id = add_request(prompts[i], sampling_params);
        results[i].prompt = prompts[i];
        request_to_index.emplace(request_id, i);
    }

    while (has_unfinished_requests())
    {
        const std::vector<LLMStepOutput> step_outputs = step();
        last_generation_profile_.add(last_step_profile());
        for (const LLMStepOutput &output : step_outputs)
        {
            auto it = request_to_index.find(output.request_id);
            if (it == request_to_index.end())
            {
                continue;
            }

            const size_t prompt_index = it->second;
            CompletionOutput &result = results[prompt_index];
            result.text = output.text;
            result.token_ids = output.token_ids;
            result.finished = output.finished;
            result.finish_reason = output.finish_reason;

            if (callback)
            {
                CompletionStreamOutput stream_output;
                static_cast<CompletionOutput &>(stream_output) = result;
                stream_output.prompt_index = prompt_index;
                stream_output.delta_text = output.delta_text;
                stream_output.token_id = output.token_id;
                callback(stream_output);
            }
        }
    }

    return results;
}

CompletionOutput LLM::generate(const std::string &prompt, const LLMSamplingParams &sampling_params,
                               CompletionStreamCallback callback)
{
    std::vector<CompletionOutput> outputs =
        generate(std::vector<std::string>{prompt}, sampling_params, std::move(callback));
    return outputs.empty() ? CompletionOutput{} : std::move(outputs.front());
}

uint64_t LLM::add_request(const std::string &prompt, const LLMSamplingParams &sampling_params)
{
    if (engine_ == nullptr)
    {
        throw std::runtime_error("LLM::add_request: engine is not initialized.");
    }
    return engine_->add_request(prompt, sampling_params);
}

bool LLM::has_unfinished_requests() const
{
    return engine_ != nullptr && engine_->has_unfinished_requests();
}

std::vector<LLMStepOutput> LLM::step()
{
    if (engine_ == nullptr)
    {
        throw std::runtime_error("LLM::step: engine is not initialized.");
    }

    const std::vector<UserOutput> user_outputs = engine_->step();
    last_step_profile_ = engine_->last_step_profile();

    std::vector<LLMStepOutput> outputs;
    outputs.reserve(user_outputs.size());
    for (const UserOutput &output : user_outputs)
    {
        if (!output.error_message.empty())
        {
            throw std::runtime_error("LLM::step: request " + std::to_string(output.internal_id) +
                                     " failed: " + output.error_message);
        }

        LLMStepOutput step_output;
        step_output.request_id = output.internal_id;
        step_output.delta_text = output.delta_text;
        step_output.text = output.text;
        step_output.token_ids = output.generated_token_ids;
        step_output.token_id = step_output.token_ids.empty() ? -1 : step_output.token_ids.back();
        step_output.finished = output.is_finished;
        step_output.finish_reason = output.finish_reason;
        outputs.push_back(std::move(step_output));
    }
    return outputs;
}

} // namespace tiny_llm
