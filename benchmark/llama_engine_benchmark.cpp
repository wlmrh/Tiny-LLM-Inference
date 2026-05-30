#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/runtime/generation_config.h"
#include "tiny_llm/runtime/llm.h"
#include "tiny_llm/runtime/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#if TINYLLM_ENABLE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    tiny_llm::ParallelConfig parallel_config = tiny_llm::ParallelConfig::cpu();
    std::string device_text = "cpu";
    int32_t warmup = 1;
    int32_t repeat = 3;
    int32_t max_new_tokens = 8;
    bool json = false;
    bool profile_detail = false;
    std::vector<std::string> prompts;
    std::filesystem::path model_dir;
    int64_t prompt_tokens = -1;
    std::vector<int64_t> prompt_token_counts;
    size_t kv_num_blocks = 0;
    bool kv_num_blocks_explicit = false;
};

struct PromptTokenStats {
    int64_t total = 0;
    std::vector<int64_t> per_prompt;
};

struct SampleOutput {
    std::string prompt;
    std::string text;
    std::string generated_text;
    std::vector<int32_t> token_ids;
    bool finished = false;
    std::string finish_reason;
};

struct RepeatMetrics {
    double load_ms = 0.0;
    double total_ms = 0.0;
    double first_token_ms = 0.0;
    double prepare_inputs_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms_total = 0.0;
    double sampling_ms = 0.0;
    double embedding_ms = 0.0;
    double qkv_proj_ms = 0.0;
    double rope_ms = 0.0;
    double attention_ms = 0.0;
    double o_proj_ms = 0.0;
    double mlp_ms = 0.0;
    double norm_ms = 0.0;
    double lm_head_ms = 0.0;
    int64_t prompt_tokens = 0;
    int64_t generated_tokens = 0;
    int64_t prefill_tokens = 0;
    int64_t decode_tokens = 0;
    bool cuda_memory_available = false;
    double cuda_memory_allocated_mb = 0.0;
    double cuda_memory_reserved_mb = 0.0;
    double cuda_memory_peak_allocated_mb = 0.0;
    double cuda_memory_peak_reserved_mb = 0.0;
    std::vector<SampleOutput> samples;
};

struct CudaMemoryMetrics {
    bool available = false;
    double allocated_mb = 0.0;
    double reserved_mb = 0.0;
    double peak_allocated_mb = 0.0;
    double peak_reserved_mb = 0.0;
};

std::filesystem::path expand_user_path(const std::string& path)
{
    if (path.empty() || path[0] != '~')
    {
        return std::filesystem::path(path);
    }

    const char* home = std::getenv("HOME");
    if (home == nullptr || *home == '\0')
    {
        return std::filesystem::path(path);
    }
    if (path.size() == 1)
    {
        return std::filesystem::path(home);
    }
    if (path[1] == '/')
    {
        return std::filesystem::path(home) / path.substr(2);
    }
    return std::filesystem::path(path);
}

int32_t parse_positive_int(const char* text, const char* name)
{
    try
    {
        size_t consumed = 0;
        const long value = std::stol(text, &consumed);
        if (consumed != std::string(text).size() || value <= 0 || value > std::numeric_limits<int32_t>::max())
        {
            throw std::runtime_error("expected positive int32");
        }
        return static_cast<int32_t>(value);
    }
    catch (const std::exception& ex)
    {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text + " (" + ex.what() + ")");
    }
}

int32_t parse_non_negative_int(const char* text, const char* name)
{
    try
    {
        size_t consumed = 0;
        const long value = std::stol(text, &consumed);
        if (consumed != std::string(text).size() || value < 0 || value > std::numeric_limits<int32_t>::max())
        {
            throw std::runtime_error("expected non-negative int32");
        }
        return static_cast<int32_t>(value);
    }
    catch (const std::exception& ex)
    {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text + " (" + ex.what() + ")");
    }
}

tiny_llm::ParallelConfig parse_device(const std::string& text)
{
    if (text == "cpu")
    {
        return tiny_llm::ParallelConfig::cpu();
    }
    if (text == "cuda")
    {
        return tiny_llm::ParallelConfig::cuda(0);
    }

    const std::string prefix = "cuda:";
    if (text.rfind(prefix, 0) == 0)
    {
        const int32_t device_id = parse_non_negative_int(text.c_str() + prefix.size(), "cuda device id");
        return tiny_llm::ParallelConfig::cuda(device_id);
    }

    throw std::runtime_error("device must be cpu, cuda, or cuda:<device_id>.");
}

bool has_safetensors_weight(const std::filesystem::path& model_dir)
{
    if (std::filesystem::exists(model_dir / "model.safetensors"))
    {
        return true;
    }
    if (!std::filesystem::is_directory(model_dir))
    {
        return false;
    }
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(model_dir))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors")
        {
            return true;
        }
    }
    return false;
}

void validate_model_dir(const std::filesystem::path& model_dir)
{
    if (!std::filesystem::is_directory(model_dir))
    {
        throw std::runtime_error("model_dir is not a directory: " + model_dir.string());
    }
    if (!std::filesystem::exists(model_dir / "config.json"))
    {
        throw std::runtime_error("model_dir must contain config.json: " + model_dir.string());
    }
    if (!std::filesystem::exists(model_dir / "tokenizer.json")
        && !std::filesystem::exists(model_dir / "tokenizer.model"))
    {
        throw std::runtime_error("model_dir must contain tokenizer.json or tokenizer.model: " + model_dir.string());
    }
    if (!has_safetensors_weight(model_dir))
    {
        throw std::runtime_error("model_dir must contain model.safetensors or safetensors shards: " + model_dir.string());
    }
}

void print_usage(const char* argv0)
{
    std::cerr << "usage: " << argv0
              << " [--device cpu|cuda[:id]] [--warmup N] [--repeat N]"
              << " [--max-new-tokens N] [--kv-num-blocks N] [--prompt TEXT]..."
              << " [--json] [--profile-detail] <model_dir>\n";
}

Options parse_args(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(std::string(name) + " requires a value.");
            }
            return argv[++i];
        };

        if (arg == "--device")
        {
            options.device_text = require_value("--device");
            options.parallel_config = parse_device(options.device_text);
        }
        else if (arg == "--warmup")
        {
            options.warmup = parse_non_negative_int(require_value("--warmup"), "warmup");
        }
        else if (arg == "--repeat")
        {
            options.repeat = parse_positive_int(require_value("--repeat"), "repeat");
        }
        else if (arg == "--max-new-tokens")
        {
            options.max_new_tokens = parse_positive_int(require_value("--max-new-tokens"), "max-new-tokens");
        }
        else if (arg == "--kv-num-blocks")
        {
            options.kv_num_blocks = static_cast<size_t>(parse_positive_int(require_value("--kv-num-blocks"), "kv-num-blocks"));
            options.kv_num_blocks_explicit = true;
        }
        else if (arg == "--prompt")
        {
            options.prompts.push_back(require_value("--prompt"));
        }
        else if (arg == "--json")
        {
            options.json = true;
        }
        else if (arg == "--profile-detail")
        {
            options.profile_detail = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (!arg.empty() && arg[0] == '-')
        {
            throw std::runtime_error("unknown option: " + arg);
        }
        else
        {
            if (!options.model_dir.empty())
            {
                throw std::runtime_error("multiple model_dir arguments were provided.");
            }
            options.model_dir = expand_user_path(arg);
        }
    }

    if (options.model_dir.empty())
    {
        throw std::runtime_error("model_dir is required.");
    }
    if (options.prompts.empty())
    {
        options.prompts = {"hello", "tiny llm inference"};
    }
    options.parallel_config.validate();
#if !TINYLLM_ENABLE_CUDA
    if (options.parallel_config.is_cuda())
    {
        throw std::runtime_error("--device cuda requires a CUDA build.");
    }
#endif
    validate_model_dir(options.model_dir);
    return options;
}

double bytes_to_mb(int64_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

CudaMemoryMetrics current_cuda_memory(const tiny_llm::ParallelConfig& parallel_config)
{
    CudaMemoryMetrics metrics;
#if TINYLLM_ENABLE_CUDA
    if (!parallel_config.is_cuda())
    {
        return metrics;
    }
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
        static_cast<c10::DeviceIndex>(parallel_config.device_id()));
    constexpr size_t kAggregate = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
    metrics.available = true;
    metrics.allocated_mb = bytes_to_mb(stats.allocated_bytes[kAggregate].current);
    metrics.reserved_mb = bytes_to_mb(stats.reserved_bytes[kAggregate].current);
    metrics.peak_allocated_mb = bytes_to_mb(stats.allocated_bytes[kAggregate].peak);
    metrics.peak_reserved_mb = bytes_to_mb(stats.reserved_bytes[kAggregate].peak);
#else
    (void)parallel_config;
#endif
    return metrics;
}

void reset_cuda_peak_memory(const tiny_llm::ParallelConfig& parallel_config)
{
#if TINYLLM_ENABLE_CUDA
    if (parallel_config.is_cuda())
    {
        c10::cuda::CUDACachingAllocator::resetPeakStats(
            static_cast<c10::DeviceIndex>(parallel_config.device_id()));
    }
#else
    (void)parallel_config;
#endif
}

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool env_flag_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr)
    {
        return false;
    }
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

double mean(const std::vector<double>& values)
{
    if (values.empty())
    {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

PromptTokenStats count_prompt_tokens(const std::filesystem::path& model_dir, const std::vector<std::string>& prompts)
{
    tiny_llm::HFLlamaTokenizer tokenizer = tiny_llm::HFLlamaTokenizer::from_model_dir(model_dir.string());
    PromptTokenStats stats;
    stats.per_prompt.reserve(prompts.size());
    for (const std::string& prompt : prompts)
    {
        const int64_t count = static_cast<int64_t>(tokenizer.encode(prompt).size());
        stats.total += count;
        stats.per_prompt.push_back(count);
    }
    return stats;
}

size_t estimate_kv_num_blocks(const std::vector<int64_t>& prompt_token_counts,
                              int32_t max_new_tokens,
                              int32_t block_size_tokens,
                              int32_t num_layers)
{
    if (block_size_tokens <= 0 || num_layers <= 0)
    {
        throw std::runtime_error("invalid dimensions for KV block estimate.");
    }
    size_t required = 0;
    for (int64_t prompt_tokens : prompt_token_counts)
    {
        const int64_t total_tokens = prompt_tokens + max_new_tokens;
        const int64_t blocks_per_layer =
            (total_tokens + block_size_tokens - 1) / block_size_tokens;
        required += static_cast<size_t>(blocks_per_layer) * static_cast<size_t>(num_layers);
    }
    const size_t with_slack = (required * 6 + 4) / 5;
    return std::max<size_t>(256, with_slack);
}

RepeatMetrics run_once(const Options& options, tiny_llm::LLM& llm, double load_ms, bool measure)
{
    RepeatMetrics metrics;

    const tiny_llm::GenerationConfig generation_config =
        tiny_llm::load_generation_config_from_dir(options.model_dir.string());
    tiny_llm::UserSamplingParams sampling_params;
    sampling_params.temperature = 0.0f;
    sampling_params.top_p = 1.0f;
    sampling_params.top_k = 0;
    sampling_params.repetition_penalty = generation_config.repetition_penalty;
    sampling_params.max_tokens = options.max_new_tokens;

    bool saw_first_token = false;
    Clock::time_point first_token_time{};
    if (measure)
    {
        reset_cuda_peak_memory(options.parallel_config);
    }
    const auto generation_start = Clock::now();
    const std::vector<tiny_llm::CompletionOutput> outputs =
        llm.generate_stream(options.prompts, sampling_params, [&](const tiny_llm::CompletionStreamOutput&) {
            if (!saw_first_token)
            {
                saw_first_token = true;
                first_token_time = Clock::now();
            }
        });
    const auto generation_end = Clock::now();

    if (!measure)
    {
        return metrics;
    }

    metrics.load_ms = load_ms;
    metrics.total_ms = elapsed_ms(generation_start, generation_end);
    metrics.first_token_ms = saw_first_token ? elapsed_ms(generation_start, first_token_time) : 0.0;
    const tiny_llm::RuntimeProfilingStats& profile = llm.last_generation_profile();
    metrics.prepare_inputs_ms = profile.prepare_inputs_ms;
    metrics.prefill_ms = profile.prefill_ms;
    metrics.decode_ms_total = profile.decode_ms_total;
    metrics.sampling_ms = profile.sampling_ms;
    metrics.embedding_ms = profile.embedding_ms;
    metrics.qkv_proj_ms = profile.qkv_proj_ms;
    metrics.rope_ms = profile.rope_ms;
    metrics.attention_ms = profile.attention_ms;
    metrics.o_proj_ms = profile.o_proj_ms;
    metrics.mlp_ms = profile.mlp_ms;
    metrics.norm_ms = profile.norm_ms;
    metrics.lm_head_ms = profile.lm_head_ms;
    const CudaMemoryMetrics memory = current_cuda_memory(options.parallel_config);
    metrics.cuda_memory_available = memory.available;
    metrics.cuda_memory_allocated_mb = memory.allocated_mb;
    metrics.cuda_memory_reserved_mb = memory.reserved_mb;
    metrics.cuda_memory_peak_allocated_mb = memory.peak_allocated_mb;
    metrics.cuda_memory_peak_reserved_mb = memory.peak_reserved_mb;
    metrics.prefill_tokens = profile.prefill_tokens;
    metrics.decode_tokens = profile.decode_tokens;
    metrics.prompt_tokens = options.prompt_tokens;
    metrics.generated_tokens = 0;
    metrics.samples.reserve(outputs.size());
    for (const tiny_llm::CompletionOutput& output : outputs)
    {
        metrics.generated_tokens += static_cast<int64_t>(output.token_ids.size());
        SampleOutput sample;
        sample.prompt = output.prompt;
        sample.text = output.text;
        sample.generated_text = output.text;
        if (output.text.rfind(output.prompt, 0) == 0)
        {
            sample.generated_text = output.text.substr(output.prompt.size());
        }
        sample.token_ids = output.token_ids;
        sample.finished = output.finished;
        sample.finish_reason = output.finish_reason;
        metrics.samples.push_back(std::move(sample));
    }
    return metrics;
}

std::string json_escape(const std::string& text)
{
    std::ostringstream out;
    for (unsigned char ch : text)
    {
        switch (ch)
        {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20)
            {
                const char* hex = "0123456789abcdef";
                out << "\\u00" << hex[(ch >> 4) & 0x0f] << hex[ch & 0x0f];
            }
            else
            {
                out << static_cast<char>(ch);
            }
            break;
        }
    }
    return out.str();
}


void print_json_int_array(const std::vector<int32_t>& values)
{
    std::cout << "[";
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ",";
        }
        std::cout << values[i];
    }
    std::cout << "]";
}

void print_samples_json(const std::vector<SampleOutput>& samples)
{
    std::cout << "[";
    for (size_t i = 0; i < samples.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ",";
        }
        const SampleOutput& sample = samples[i];
        std::cout << "{";
        std::cout << "\"prompt\":\"" << json_escape(sample.prompt) << "\",";
        std::cout << "\"output_text\":\"" << json_escape(sample.text) << "\",";
        std::cout << "\"generated_text\":\"" << json_escape(sample.generated_text) << "\",";
        std::cout << "\"token_ids\":";
        print_json_int_array(sample.token_ids);
        std::cout << ",\"finished\":" << (sample.finished ? "true" : "false") << ",";
        std::cout << "\"finish_reason\":\"" << json_escape(sample.finish_reason) << "\"";
        std::cout << "}";
    }
    std::cout << "]";
}

void print_summary(const Options& options, const std::vector<RepeatMetrics>& repeats)
{
    std::vector<double> load_ms;
    std::vector<double> total_ms;
    std::vector<double> first_token_ms;
    std::vector<double> prepare_inputs_ms;
    std::vector<double> prefill_ms;
    std::vector<double> decode_ms_total;
    std::vector<double> decode_ms_per_token;
    std::vector<double> sampling_ms;
    std::vector<double> embedding_ms;
    std::vector<double> qkv_proj_ms;
    std::vector<double> rope_ms;
    std::vector<double> attention_ms;
    std::vector<double> o_proj_ms;
    std::vector<double> mlp_ms;
    std::vector<double> norm_ms;
    std::vector<double> lm_head_ms;
    std::vector<double> cuda_memory_allocated_mb;
    std::vector<double> cuda_memory_reserved_mb;
    std::vector<double> cuda_memory_peak_allocated_mb;
    std::vector<double> cuda_memory_peak_reserved_mb;
    load_ms.reserve(repeats.size());
    total_ms.reserve(repeats.size());
    first_token_ms.reserve(repeats.size());
    prepare_inputs_ms.reserve(repeats.size());
    prefill_ms.reserve(repeats.size());
    decode_ms_total.reserve(repeats.size());
    decode_ms_per_token.reserve(repeats.size());
    sampling_ms.reserve(repeats.size());
    embedding_ms.reserve(repeats.size());
    qkv_proj_ms.reserve(repeats.size());
    rope_ms.reserve(repeats.size());
    attention_ms.reserve(repeats.size());
    o_proj_ms.reserve(repeats.size());
    mlp_ms.reserve(repeats.size());
    norm_ms.reserve(repeats.size());
    lm_head_ms.reserve(repeats.size());
    cuda_memory_allocated_mb.reserve(repeats.size());
    cuda_memory_reserved_mb.reserve(repeats.size());
    cuda_memory_peak_allocated_mb.reserve(repeats.size());
    cuda_memory_peak_reserved_mb.reserve(repeats.size());
    int64_t generated_tokens = 0;
    int64_t prompt_tokens = -1;
    int64_t prefill_tokens = 0;
    int64_t decode_tokens = 0;
    for (const RepeatMetrics& metrics : repeats)
    {
        load_ms.push_back(metrics.load_ms);
        total_ms.push_back(metrics.total_ms);
        first_token_ms.push_back(metrics.first_token_ms);
        prepare_inputs_ms.push_back(metrics.prepare_inputs_ms);
        prefill_ms.push_back(metrics.prefill_ms);
        decode_ms_total.push_back(metrics.decode_ms_total);
        decode_ms_per_token.push_back(metrics.decode_tokens > 0
            ? metrics.decode_ms_total / static_cast<double>(metrics.decode_tokens)
            : 0.0);
        sampling_ms.push_back(metrics.sampling_ms);
        embedding_ms.push_back(metrics.embedding_ms);
        qkv_proj_ms.push_back(metrics.qkv_proj_ms);
        rope_ms.push_back(metrics.rope_ms);
        attention_ms.push_back(metrics.attention_ms);
        o_proj_ms.push_back(metrics.o_proj_ms);
        mlp_ms.push_back(metrics.mlp_ms);
        norm_ms.push_back(metrics.norm_ms);
        lm_head_ms.push_back(metrics.lm_head_ms);
        if (metrics.cuda_memory_available)
        {
            cuda_memory_allocated_mb.push_back(metrics.cuda_memory_allocated_mb);
            cuda_memory_reserved_mb.push_back(metrics.cuda_memory_reserved_mb);
            cuda_memory_peak_allocated_mb.push_back(metrics.cuda_memory_peak_allocated_mb);
            cuda_memory_peak_reserved_mb.push_back(metrics.cuda_memory_peak_reserved_mb);
        }
        generated_tokens += metrics.generated_tokens;
        prefill_tokens += metrics.prefill_tokens;
        decode_tokens += metrics.decode_tokens;
        prompt_tokens = metrics.prompt_tokens;
    }

    const double avg_load_ms = mean(load_ms);
    const double avg_total_ms = mean(total_ms);
    const double avg_first_ms = mean(first_token_ms);
    const double avg_prepare_inputs_ms = mean(prepare_inputs_ms);
    const double avg_prefill_ms = mean(prefill_ms);
    const double avg_decode_ms_total = mean(decode_ms_total);
    const double avg_decode_ms_per_token = mean(decode_ms_per_token);
    const double avg_sampling_ms = mean(sampling_ms);
    const double avg_embedding_ms = mean(embedding_ms);
    const double avg_qkv_proj_ms = mean(qkv_proj_ms);
    const double avg_rope_ms = mean(rope_ms);
    const double avg_attention_ms = mean(attention_ms);
    const double avg_o_proj_ms = mean(o_proj_ms);
    const double avg_mlp_ms = mean(mlp_ms);
    const double avg_norm_ms = mean(norm_ms);
    const double avg_lm_head_ms = mean(lm_head_ms);
    const double avg_cuda_memory_allocated_mb = mean(cuda_memory_allocated_mb);
    const double avg_cuda_memory_reserved_mb = mean(cuda_memory_reserved_mb);
    const double avg_cuda_memory_peak_allocated_mb = mean(cuda_memory_peak_allocated_mb);
    const double avg_cuda_memory_peak_reserved_mb = mean(cuda_memory_peak_reserved_mb);
    const bool has_cuda_memory = !cuda_memory_allocated_mb.empty();
    const double avg_generated_tokens = repeats.empty() ? 0.0 : static_cast<double>(generated_tokens) / repeats.size();
    const double e2e_tokens_per_s = avg_total_ms > 0.0 ? avg_generated_tokens / (avg_total_ms / 1000.0) : 0.0;
    const double decode_ms = std::max(0.0, avg_total_ms - avg_first_ms);
    const double generated_decode_tokens = std::max(0.0, avg_generated_tokens - static_cast<double>(options.prompts.size()));
    const double decode_tokens_per_s = decode_ms > 0.0 ? generated_decode_tokens / (decode_ms / 1000.0) : 0.0;

    std::cout << "llama_engine_benchmark\n";
    std::cout << "  model: " << options.model_dir.string() << "\n";
    std::cout << "  device: " << options.device_text << "\n";
    std::cout << "  prompts: " << options.prompts.size() << ", warmup: " << options.warmup
              << ", repeat: " << options.repeat << ", max_new_tokens: " << options.max_new_tokens
              << ", profile_detail: " << (options.profile_detail ? "on" : "off")
              << ", kv_num_blocks: " << options.kv_num_blocks << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  avg_load_init_ms: " << avg_load_ms << "\n";
    std::cout << "  avg_total_latency_ms: " << avg_total_ms << "\n";
    std::cout << "  avg_first_token_latency_ms: " << avg_first_ms << "\n";
    std::cout << "  avg_generated_tokens: " << avg_generated_tokens << "\n";
    std::cout << "  total_generated_tokens: " << generated_tokens << "\n";
    std::cout << "  prepare_inputs_ms: " << avg_prepare_inputs_ms << "\n";
    std::cout << "  prefill_ms: " << avg_prefill_ms << "\n";
    std::cout << "  decode_ms_total: " << avg_decode_ms_total << "\n";
    std::cout << "  decode_ms_per_token: " << avg_decode_ms_per_token << "\n";
    std::cout << "  sampling_ms: " << avg_sampling_ms << "\n";
    if (has_cuda_memory)
    {
        std::cout << "  cuda_memory_allocated_mb: " << avg_cuda_memory_allocated_mb << "\n";
        std::cout << "  cuda_memory_reserved_mb: " << avg_cuda_memory_reserved_mb << "\n";
        std::cout << "  cuda_memory_peak_allocated_mb: " << avg_cuda_memory_peak_allocated_mb << "\n";
        std::cout << "  cuda_memory_peak_reserved_mb: " << avg_cuda_memory_peak_reserved_mb << "\n";
    }
    if (options.profile_detail)
    {
        std::cout << "  embedding_ms: " << avg_embedding_ms << "\n";
        std::cout << "  qkv_proj_ms: " << avg_qkv_proj_ms << "\n";
        std::cout << "  rope_ms: " << avg_rope_ms << "\n";
        std::cout << "  attention_ms: " << avg_attention_ms << "\n";
        std::cout << "  o_proj_ms: " << avg_o_proj_ms << "\n";
        std::cout << "  mlp_ms: " << avg_mlp_ms << "\n";
        std::cout << "  norm_ms: " << avg_norm_ms << "\n";
        std::cout << "  lm_head_ms: " << avg_lm_head_ms << "\n";
    }
    std::cout << "  avg_prefill_tokens: " << (repeats.empty() ? 0.0 : static_cast<double>(prefill_tokens) / repeats.size()) << "\n";
    std::cout << "  avg_decode_tokens: " << (repeats.empty() ? 0.0 : static_cast<double>(decode_tokens) / repeats.size()) << "\n";
    if (prompt_tokens >= 0)
    {
        std::cout << "  prompt_tokens: " << prompt_tokens << "\n";
    }
    else
    {
        std::cout << "  prompt_tokens: unavailable\n";
    }
    std::cout << "  end_to_end_tokens_per_s: " << e2e_tokens_per_s << "\n";
    std::cout << "  decode_tokens_per_s: " << decode_tokens_per_s << "\n";
    std::cout << "  repeat_total_latency_ms: [";
    for (size_t i = 0; i < repeats.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ", ";
        }
        std::cout << repeats[i].total_ms;
    }
    std::cout << "]\n";
    std::cout << "  repeat_load_init_ms: [";
    for (size_t i = 0; i < repeats.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ", ";
        }
        std::cout << repeats[i].load_ms;
    }
    std::cout << "]\n";
    std::cout << "  repeat_prepare_inputs_ms: [";
    for (size_t i = 0; i < repeats.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ", ";
        }
        std::cout << repeats[i].prepare_inputs_ms;
    }
    std::cout << "]\n";
    std::cout << "  repeat_prefill_ms: [";
    for (size_t i = 0; i < repeats.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ", ";
        }
        std::cout << repeats[i].prefill_ms;
    }
    std::cout << "]\n";
    std::cout << "  repeat_decode_ms_total: [";
    for (size_t i = 0; i < repeats.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ", ";
        }
        std::cout << repeats[i].decode_ms_total;
    }
    std::cout << "]\n";
    std::cout << "  repeat_sampling_ms: [";
    for (size_t i = 0; i < repeats.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ", ";
        }
        std::cout << repeats[i].sampling_ms;
    }
    std::cout << "]\n";
    if (!repeats.empty())
    {
        std::cout << "  samples:\n";
        for (size_t i = 0; i < repeats.front().samples.size(); ++i)
        {
            const SampleOutput& sample = repeats.front().samples[i];
            std::cout << "    [" << i << "] prompt: " << sample.prompt << "\n";
            std::cout << "    [" << i << "] output_text: " << sample.text << "\n";
            std::cout << "    [" << i << "] generated_text: " << sample.generated_text << "\n";
            std::cout << "    [" << i << "] finish_reason: " << sample.finish_reason << "\n";
        }
    }

    if (!options.json)
    {
        return;
    }

    std::cout << "{";
    std::cout << "\"benchmark\":\"llama_engine_benchmark\",";
    std::cout << "\"backend\":\"tinyllm\",";
    std::cout << "\"model\":\"" << json_escape(options.model_dir.string()) << "\",";
    std::cout << "\"device\":\"" << json_escape(options.device_text) << "\",";
    std::cout << "\"prompt_count\":" << options.prompts.size() << ",";
    std::cout << "\"prompt_tokens\":" << prompt_tokens << ",";
    std::cout << "\"warmup\":" << options.warmup << ",";
    std::cout << "\"repeat\":" << options.repeat << ",";
    std::cout << "\"max_new_tokens\":" << options.max_new_tokens << ",";
    std::cout << "\"profile_detail\":" << (options.profile_detail ? "true" : "false") << ",";
    std::cout << "\"kv_num_blocks\":" << options.kv_num_blocks << ",";
    std::cout << "\"avg_load_init_ms\":" << avg_load_ms << ",";
    std::cout << "\"avg_total_latency_ms\":" << avg_total_ms << ",";
    std::cout << "\"avg_first_token_latency_ms\":" << avg_first_ms << ",";
    std::cout << "\"avg_generated_tokens\":" << avg_generated_tokens << ",";
    std::cout << "\"total_generated_tokens\":" << generated_tokens << ",";
    std::cout << "\"prepare_inputs_ms\":" << avg_prepare_inputs_ms << ",";
    std::cout << "\"prefill_ms\":" << avg_prefill_ms << ",";
    std::cout << "\"decode_ms_total\":" << avg_decode_ms_total << ",";
    std::cout << "\"decode_ms_per_token\":" << avg_decode_ms_per_token << ",";
    std::cout << "\"sampling_ms\":" << avg_sampling_ms << ",";
    std::cout << "\"embedding_ms\":" << avg_embedding_ms << ",";
    std::cout << "\"qkv_proj_ms\":" << avg_qkv_proj_ms << ",";
    std::cout << "\"rope_ms\":" << avg_rope_ms << ",";
    std::cout << "\"attention_ms\":" << avg_attention_ms << ",";
    std::cout << "\"o_proj_ms\":" << avg_o_proj_ms << ",";
    std::cout << "\"mlp_ms\":" << avg_mlp_ms << ",";
    std::cout << "\"norm_ms\":" << avg_norm_ms << ",";
    std::cout << "\"lm_head_ms\":" << avg_lm_head_ms << ",";
    std::cout << "\"cuda_memory_allocated_mb\":" << avg_cuda_memory_allocated_mb << ",";
    std::cout << "\"cuda_memory_reserved_mb\":" << avg_cuda_memory_reserved_mb << ",";
    std::cout << "\"cuda_memory_peak_allocated_mb\":" << avg_cuda_memory_peak_allocated_mb << ",";
    std::cout << "\"cuda_memory_peak_reserved_mb\":" << avg_cuda_memory_peak_reserved_mb << ",";
    std::cout << "\"avg_prefill_tokens\":" << (repeats.empty() ? 0.0 : static_cast<double>(prefill_tokens) / repeats.size()) << ",";
    std::cout << "\"avg_decode_tokens\":" << (repeats.empty() ? 0.0 : static_cast<double>(decode_tokens) / repeats.size()) << ",";
    std::cout << "\"end_to_end_tokens_per_s\":" << e2e_tokens_per_s << ",";
    std::cout << "\"decode_tokens_per_s\":" << decode_tokens_per_s << ",";
    std::cout << "\"repeat_total_latency_ms\":[";
    for (size_t i = 0; i < repeats.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ",";
        }
        std::cout << repeats[i].total_ms;
    }
    std::cout << "],";
    std::cout << "\"repeat_load_init_ms\":[";
    for (size_t i = 0; i < repeats.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ",";
        }
        std::cout << repeats[i].load_ms;
    }
    std::cout << "],";
    std::cout << "\"repeat_prepare_inputs_ms\":[";
    for (size_t i = 0; i < repeats.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ",";
        }
        std::cout << repeats[i].prepare_inputs_ms;
    }
    std::cout << "],";
    std::cout << "\"repeat_prefill_ms\":[";
    for (size_t i = 0; i < repeats.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ",";
        }
        std::cout << repeats[i].prefill_ms;
    }
    std::cout << "],";
    std::cout << "\"repeat_decode_ms_total\":[";
    for (size_t i = 0; i < repeats.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ",";
        }
        std::cout << repeats[i].decode_ms_total;
    }
    std::cout << "],";
    std::cout << "\"samples\":";
    if (!repeats.empty())
    {
        print_samples_json(repeats.front().samples);
    }
    else
    {
        std::cout << "[]";
    }
    std::cout << ",";
    std::cout << "\"repeat_sampling_ms\":[";
    for (size_t i = 0; i < repeats.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ",";
        }
        std::cout << repeats[i].sampling_ms;
    }
    std::cout << "]}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        Options options = parse_args(argc, argv);
        options.profile_detail = options.profile_detail || env_flag_enabled("TINYLLM_PROFILE_DETAIL");
        if (options.profile_detail)
        {
            setenv("TINYLLM_PROFILE_DETAIL", "1", 1);
        }
        const PromptTokenStats token_stats = count_prompt_tokens(options.model_dir, options.prompts);
        options.prompt_tokens = token_stats.total;
        options.prompt_token_counts = token_stats.per_prompt;
        const tiny_llm::LlamaConfig hf_config =
            tiny_llm::HFLlamaConfigLoader::load_from_dir(options.model_dir.string());
        constexpr int32_t kBlockSizeTokens = 16;
        if (!options.kv_num_blocks_explicit)
        {
            options.kv_num_blocks = estimate_kv_num_blocks(
                options.prompt_token_counts,
                options.max_new_tokens,
                kBlockSizeTokens,
                hf_config.num_hidden_layers);
        }

        const auto load_start = Clock::now();
        tiny_llm::LLMOptions llm_options(options.model_dir.string(), options.parallel_config);
        llm_options.max_tokens = options.max_new_tokens;
        llm_options.block_size_tokens = kBlockSizeTokens;
        llm_options.kv_num_blocks = options.kv_num_blocks;
        tiny_llm::LLM llm(llm_options);
        const double load_ms = elapsed_ms(load_start, Clock::now());

        for (int32_t i = 0; i < options.warmup; ++i)
        {
            (void)run_once(options, llm, load_ms, false);
        }

        std::vector<RepeatMetrics> repeats;
        repeats.reserve(static_cast<size_t>(options.repeat));
        for (int32_t i = 0; i < options.repeat; ++i)
        {
            repeats.push_back(run_once(options, llm, load_ms, true));
        }
        print_summary(options, repeats);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "llama_engine_benchmark failed: " << ex.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
