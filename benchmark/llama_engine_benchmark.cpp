#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/runtime/generation_config.h"
#include "tiny_llm/runtime/llm.h"
#include "tiny_llm/runtime/tokenizer.h"

#include "progress_guard.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#if TINYLLM_ENABLE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace
{

using Clock = std::chrono::steady_clock;

struct Options
{
    tiny_llm::ParallelConfig parallel_config = tiny_llm::ParallelConfig::cpu();
    std::string device_text = "cpu";
    tiny_llm::RuntimeDType compute_dtype = tiny_llm::RuntimeDType::kFloat32;
    tiny_llm::RuntimeDType kv_cache_dtype = tiny_llm::RuntimeDType::kFloat32;
    int32_t warmup = 1;
    int32_t repeat = 3;
    int32_t max_new_tokens = 8;
    bool max_new_tokens_explicit = false;
    bool ignore_eos = false;
    bool ignore_eos_explicit = false;
    bool json = false;
    bool profile_detail = false;
    std::vector<std::string> prompts;
    std::vector<std::string> request_ids;
    std::vector<double> arrival_ms;
    std::string traffic_mode = "offline";
    std::filesystem::path model_dir;
    std::filesystem::path workload_jsonl;
    std::filesystem::path events_jsonl;
    int64_t prompt_tokens = -1;
    std::vector<int64_t> prompt_token_counts;
    size_t kv_num_blocks = 0;
    bool kv_num_blocks_explicit = false;
    int32_t max_num_batched_tokens = 0;
    bool max_num_batched_tokens_explicit = false;
    int32_t max_num_batched_token_cap = 4096;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int32_t top_k = 0;
    float repetition_penalty = 1.0f;
    bool repetition_penalty_explicit = false;
    uint64_t seed = 0;
    std::vector<int32_t> stop_token_ids;
};

struct PromptTokenStats
{
    int64_t total = 0;
    std::vector<int64_t> per_prompt;
};

struct SampleOutput
{
    std::string request_id;
    std::string prompt;
    std::string text;
    std::string generated_text;
    std::vector<int32_t> token_ids;
    bool finished = false;
    std::string finish_reason;
};

struct TokenTraceEvent
{
    int32_t token_index = 0;
    int32_t token_id = -1;
    double time_ms = 0.0;
    std::string delta_text;
    bool finished = false;
};

struct RequestTrace
{
    std::string request_id;
    size_t prompt_index = 0;
    double submit_ms = 0.0;
    double admit_ms = 0.0;
    double first_token_ms = -1.0;
    double finish_ms = -1.0;
    int64_t prompt_tokens = 0;
    int64_t generated_tokens = 0;
    std::string finish_reason;
    std::vector<TokenTraceEvent> tokens;
};

struct RepeatMetrics
{
    double load_ms = 0.0;
    double total_ms = 0.0;
    double first_token_ms = 0.0;
    double prepare_inputs_ms = 0.0;
    double model_ms_total = 0.0;
    double prefill_ms = 0.0;
    double decode_ms_total = 0.0;
    double mixed_model_ms = 0.0;
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
    int64_t scheduled_requests = 0;
    int64_t scheduled_tokens = 0;
    int64_t prefill_requests = 0;
    int64_t decode_requests = 0;
    int64_t max_context_len = 0;
    int64_t profiled_steps = 0;
    bool cuda_memory_available = false;
    double cuda_memory_allocated_mb = 0.0;
    double cuda_memory_reserved_mb = 0.0;
    double cuda_memory_peak_allocated_mb = 0.0;
    double cuda_memory_peak_reserved_mb = 0.0;
    std::vector<SampleOutput> samples;
    std::vector<RequestTrace> request_traces;
};

struct CudaMemoryMetrics
{
    bool available = false;
    double allocated_mb = 0.0;
    double reserved_mb = 0.0;
    double peak_allocated_mb = 0.0;
    double peak_reserved_mb = 0.0;
};

std::filesystem::path expand_user_path(const std::string &path)
{
    if (path.empty() || path[0] != '~')
    {
        return std::filesystem::path(path);
    }

    const char *home = std::getenv("HOME");
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

int32_t parse_positive_int(const char *text, const char *name)
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
    catch (const std::exception &ex)
    {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text + " (" + ex.what() + ")");
    }
}

int32_t parse_non_negative_int(const char *text, const char *name)
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
    catch (const std::exception &ex)
    {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text + " (" + ex.what() + ")");
    }
}

float parse_float(const char *text, const char *name)
{
    try
    {
        size_t consumed = 0;
        const float value = std::stof(text, &consumed);
        if (consumed != std::string(text).size())
        {
            throw std::runtime_error("expected float");
        }
        return value;
    }
    catch (const std::exception &ex)
    {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text + " (" + ex.what() + ")");
    }
}

uint64_t parse_uint64(const char *text, const char *name)
{
    try
    {
        size_t consumed = 0;
        const unsigned long long value = std::stoull(text, &consumed);
        if (consumed != std::string(text).size())
        {
            throw std::runtime_error("expected uint64");
        }
        return static_cast<uint64_t>(value);
    }
    catch (const std::exception &ex)
    {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text + " (" + ex.what() + ")");
    }
}

tiny_llm::ParallelConfig parse_device(const std::string &text)
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

std::string json_escape(const std::string &text);

size_t find_json_value(const std::string &line, const std::string &key)
{
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = line.find(needle);
    if (key_pos == std::string::npos)
    {
        return std::string::npos;
    }
    size_t colon = line.find(':', key_pos + needle.size());
    if (colon == std::string::npos)
    {
        throw std::runtime_error("malformed workload JSON for key: " + key);
    }
    ++colon;
    while (colon < line.size() && std::isspace(static_cast<unsigned char>(line[colon])))
    {
        ++colon;
    }
    return colon;
}

int hex_digit(char ch)
{
    if (ch >= '0' && ch <= '9')
    {
        return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f')
    {
        return ch - 'a' + 10;
    }
    if (ch >= 'A' && ch <= 'F')
    {
        return ch - 'A' + 10;
    }
    return -1;
}

void append_utf8(std::string &out, uint32_t codepoint)
{
    if (codepoint <= 0x7f)
    {
        out.push_back(static_cast<char>(codepoint));
    }
    else if (codepoint <= 0x7ff)
    {
        out.push_back(static_cast<char>(0xc0 | ((codepoint >> 6) & 0x1f)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    else
    {
        out.push_back(static_cast<char>(0xe0 | ((codepoint >> 12) & 0x0f)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

std::string parse_json_string_at(const std::string &line, size_t pos, const std::string &key)
{
    if (pos >= line.size() || line[pos] != '"')
    {
        throw std::runtime_error("workload key must be a JSON string: " + key);
    }
    std::string out;
    for (size_t i = pos + 1; i < line.size(); ++i)
    {
        const char ch = line[i];
        if (ch == '"')
        {
            return out;
        }
        if (ch != '\\')
        {
            out.push_back(ch);
            continue;
        }
        if (++i >= line.size())
        {
            throw std::runtime_error("unterminated JSON escape for key: " + key);
        }
        const char esc = line[i];
        switch (esc)
        {
        case '"':
            out.push_back('"');
            break;
        case '\\':
            out.push_back('\\');
            break;
        case '/':
            out.push_back('/');
            break;
        case 'b':
            out.push_back('\b');
            break;
        case 'f':
            out.push_back('\f');
            break;
        case 'n':
            out.push_back('\n');
            break;
        case 'r':
            out.push_back('\r');
            break;
        case 't':
            out.push_back('\t');
            break;
        case 'u':
        {
            if (i + 4 >= line.size())
            {
                throw std::runtime_error("short JSON unicode escape for key: " + key);
            }
            uint32_t codepoint = 0;
            for (int digit_index = 0; digit_index < 4; ++digit_index)
            {
                const int digit = hex_digit(line[++i]);
                if (digit < 0)
                {
                    throw std::runtime_error("invalid JSON unicode escape for key: " + key);
                }
                codepoint = (codepoint << 4) | static_cast<uint32_t>(digit);
            }
            append_utf8(out, codepoint);
            break;
        }
        default:
            throw std::runtime_error("unsupported JSON escape for key: " + key);
        }
    }
    throw std::runtime_error("unterminated JSON string for key: " + key);
}

bool json_string_field(const std::string &line, const std::string &key, std::string &value)
{
    const size_t pos = find_json_value(line, key);
    if (pos == std::string::npos)
    {
        return false;
    }
    value = parse_json_string_at(line, pos, key);
    return true;
}

bool json_int_field(const std::string &line, const std::string &key, int32_t &value)
{
    const size_t pos = find_json_value(line, key);
    if (pos == std::string::npos)
    {
        return false;
    }
    size_t consumed = 0;
    const long parsed = std::stol(line.substr(pos), &consumed);
    if (consumed == 0 || parsed < std::numeric_limits<int32_t>::min() || parsed > std::numeric_limits<int32_t>::max())
    {
        throw std::runtime_error("invalid integer workload field: " + key);
    }
    value = static_cast<int32_t>(parsed);
    return true;
}

bool json_float_field(const std::string &line, const std::string &key, float &value)
{
    const size_t pos = find_json_value(line, key);
    if (pos == std::string::npos)
    {
        return false;
    }
    size_t consumed = 0;
    value = std::stof(line.substr(pos), &consumed);
    if (consumed == 0)
    {
        throw std::runtime_error("invalid float workload field: " + key);
    }
    return true;
}

bool json_double_field(const std::string &line, const std::string &key, double &value)
{
    const size_t pos = find_json_value(line, key);
    if (pos == std::string::npos)
    {
        return false;
    }
    size_t consumed = 0;
    value = std::stod(line.substr(pos), &consumed);
    if (consumed == 0 || !std::isfinite(value))
    {
        throw std::runtime_error("invalid numeric workload field: " + key);
    }
    return true;
}

bool json_bool_field(const std::string &line, const std::string &key, bool &value)
{
    const size_t pos = find_json_value(line, key);
    if (pos == std::string::npos)
    {
        return false;
    }
    if (line.compare(pos, 4, "true") == 0)
    {
        value = true;
        return true;
    }
    if (line.compare(pos, 5, "false") == 0)
    {
        value = false;
        return true;
    }
    throw std::runtime_error("invalid boolean workload field: " + key);
}

void load_workload_jsonl(Options &options)
{
    if (options.workload_jsonl.empty())
    {
        return;
    }
    std::ifstream input(options.workload_jsonl);
    if (!input)
    {
        throw std::runtime_error("failed to open workload JSONL: " + options.workload_jsonl.string());
    }

    options.prompts.clear();
    options.request_ids.clear();
    options.arrival_ms.clear();
    std::string line;
    int64_t line_number = 0;
    int32_t workload_max_new_tokens = -1;
    bool workload_ignore_eos_set = false;
    bool workload_ignore_eos = false;
    bool workload_temperature_set = false;
    bool workload_top_p_set = false;
    bool workload_top_k_set = false;
    bool workload_repetition_penalty_set = false;
    float workload_temperature = options.temperature;
    float workload_top_p = options.top_p;
    int32_t workload_top_k = options.top_k;
    float workload_repetition_penalty = options.repetition_penalty;
    while (std::getline(input, line))
    {
        ++line_number;
        if (line.empty())
        {
            continue;
        }
        std::string prompt;
        if (!json_string_field(line, "prompt", prompt))
        {
            throw std::runtime_error("workload record missing prompt at line " + std::to_string(line_number));
        }
        std::string request_id;
        if (!json_string_field(line, "request_id", request_id))
        {
            request_id = "request-" + std::to_string(options.prompts.size());
        }
        int32_t max_new_tokens = 0;
        if (json_int_field(line, "max_new_tokens", max_new_tokens))
        {
            if (max_new_tokens <= 0)
            {
                throw std::runtime_error("workload max_new_tokens must be positive.");
            }
            if (workload_max_new_tokens >= 0 && workload_max_new_tokens != max_new_tokens)
            {
                throw std::runtime_error("mixed workload max_new_tokens values are not supported by this runner.");
            }
            workload_max_new_tokens = max_new_tokens;
        }
        if (!options.ignore_eos_explicit)
        {
            bool ignore_eos = false;
            if (json_bool_field(line, "ignore_eos", ignore_eos))
            {
                if (workload_ignore_eos_set && workload_ignore_eos != ignore_eos)
                {
                    throw std::runtime_error("mixed workload ignore_eos values are not supported by this runner.");
                }
                workload_ignore_eos = ignore_eos;
                workload_ignore_eos_set = true;
            }
        }
        if (!options.repetition_penalty_explicit)
        {
            float value = 0.0f;
            if (json_float_field(line, "repetition_penalty", value))
            {
                if (workload_repetition_penalty_set && workload_repetition_penalty != value)
                {
                    throw std::runtime_error(
                        "mixed workload repetition_penalty values are not supported by this runner.");
                }
                workload_repetition_penalty = value;
                workload_repetition_penalty_set = true;
            }
        }
        float float_value = 0.0f;
        if (json_float_field(line, "temperature", float_value))
        {
            if (workload_temperature_set && workload_temperature != float_value)
            {
                throw std::runtime_error("mixed workload temperature values are not supported by this runner.");
            }
            workload_temperature = float_value;
            workload_temperature_set = true;
        }
        if (json_float_field(line, "top_p", float_value))
        {
            if (workload_top_p_set && workload_top_p != float_value)
            {
                throw std::runtime_error("mixed workload top_p values are not supported by this runner.");
            }
            workload_top_p = float_value;
            workload_top_p_set = true;
        }
        int32_t top_k = options.top_k;
        if (json_int_field(line, "top_k", top_k))
        {
            if (workload_top_k_set && workload_top_k != top_k)
            {
                throw std::runtime_error("mixed workload top_k values are not supported by this runner.");
            }
            workload_top_k = top_k;
            workload_top_k_set = true;
        }

        double arrival_ms = 0.0;
        if (json_double_field(line, "arrival_ms", arrival_ms) && arrival_ms < 0.0)
        {
            throw std::runtime_error("workload arrival_ms must be non-negative at line " + std::to_string(line_number));
        }

        options.prompts.push_back(std::move(prompt));
        options.request_ids.push_back(std::move(request_id));
        options.arrival_ms.push_back(arrival_ms);
    }
    if (!options.max_new_tokens_explicit && workload_max_new_tokens > 0)
    {
        options.max_new_tokens = workload_max_new_tokens;
    }
    if (!options.ignore_eos_explicit && workload_ignore_eos_set)
    {
        options.ignore_eos = workload_ignore_eos;
    }
    if (workload_temperature_set)
    {
        options.temperature = workload_temperature;
    }
    if (workload_top_p_set)
    {
        options.top_p = workload_top_p;
    }
    if (workload_top_k_set)
    {
        options.top_k = workload_top_k;
    }
    if (!options.repetition_penalty_explicit && workload_repetition_penalty_set)
    {
        options.repetition_penalty = workload_repetition_penalty;
        options.repetition_penalty_explicit = true;
    }
    if (options.prompts.empty())
    {
        throw std::runtime_error("workload JSONL did not contain any requests: " + options.workload_jsonl.string());
    }

    std::vector<size_t> order(options.prompts.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t lhs, size_t rhs) { return options.arrival_ms[lhs] < options.arrival_ms[rhs]; });
    std::vector<std::string> sorted_prompts;
    std::vector<std::string> sorted_request_ids;
    std::vector<double> sorted_arrival_ms;
    sorted_prompts.reserve(order.size());
    sorted_request_ids.reserve(order.size());
    sorted_arrival_ms.reserve(order.size());
    for (size_t index : order)
    {
        sorted_prompts.push_back(std::move(options.prompts[index]));
        sorted_request_ids.push_back(std::move(options.request_ids[index]));
        sorted_arrival_ms.push_back(options.arrival_ms[index]);
    }
    options.prompts = std::move(sorted_prompts);
    options.request_ids = std::move(sorted_request_ids);
    options.arrival_ms = std::move(sorted_arrival_ms);
}

bool has_safetensors_weight(const std::filesystem::path &model_dir)
{
    if (std::filesystem::exists(model_dir / "model.safetensors"))
    {
        return true;
    }
    if (!std::filesystem::is_directory(model_dir))
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

void validate_model_dir(const std::filesystem::path &model_dir)
{
    if (!std::filesystem::is_directory(model_dir))
    {
        throw std::runtime_error("model_dir is not a directory: " + model_dir.string());
    }
    if (!std::filesystem::exists(model_dir / "config.json"))
    {
        throw std::runtime_error("model_dir must contain config.json: " + model_dir.string());
    }
    if (!std::filesystem::exists(model_dir / "tokenizer.json") &&
        !std::filesystem::exists(model_dir / "tokenizer.model"))
    {
        throw std::runtime_error("model_dir must contain tokenizer.json or tokenizer.model: " + model_dir.string());
    }
    if (!has_safetensors_weight(model_dir))
    {
        throw std::runtime_error("model_dir must contain model.safetensors or safetensors shards: " +
                                 model_dir.string());
    }
}

void print_usage(const char *argv0)
{
    std::cerr << "usage: " << argv0 << " [--device cpu|cuda[:id]] [--dtype fp32|bf16] [--kv-cache-dtype fp32|bf16]"
              << " [--warmup N] [--repeat N]"
              << " [--max-new-tokens N] [--kv-num-blocks N]"
              << " [--max-num-batched-tokens N] [--max-num-batched-token-cap N]"
              << " [--temperature F] [--top-p F] [--top-k N]"
              << " [--repetition-penalty F] [--seed N] [--stop-token-id N]..."
              << " [--prompt TEXT]... [--workload-jsonl PATH] [--events-jsonl PATH]"
              << " [--traffic-mode offline|open-loop]"
              << " [--json] [--profile-detail] [--ignore-eos] <model_dir>\n";
}

Options parse_args(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        auto require_value = [&](const char *name) -> const char *
        {
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
        else if (arg == "--dtype")
        {
            options.compute_dtype = tiny_llm::parse_runtime_dtype(require_value("--dtype"));
        }
        else if (arg == "--kv-cache-dtype")
        {
            options.kv_cache_dtype = tiny_llm::parse_runtime_dtype(require_value("--kv-cache-dtype"));
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
            options.max_new_tokens_explicit = true;
        }
        else if (arg == "--kv-num-blocks")
        {
            options.kv_num_blocks =
                static_cast<size_t>(parse_positive_int(require_value("--kv-num-blocks"), "kv-num-blocks"));
            options.kv_num_blocks_explicit = true;
        }
        else if (arg == "--max-num-batched-tokens")
        {
            options.max_num_batched_tokens =
                parse_positive_int(require_value("--max-num-batched-tokens"), "max-num-batched-tokens");
            options.max_num_batched_tokens_explicit = true;
        }
        else if (arg == "--max-num-batched-token-cap")
        {
            options.max_num_batched_token_cap =
                parse_positive_int(require_value("--max-num-batched-token-cap"), "max-num-batched-token-cap");
        }
        else if (arg == "--prompt")
        {
            options.prompts.push_back(require_value("--prompt"));
        }
        else if (arg == "--workload-jsonl")
        {
            options.workload_jsonl = expand_user_path(require_value("--workload-jsonl"));
        }
        else if (arg == "--events-jsonl")
        {
            options.events_jsonl = expand_user_path(require_value("--events-jsonl"));
        }
        else if (arg == "--traffic-mode")
        {
            options.traffic_mode = require_value("--traffic-mode");
            if (options.traffic_mode != "offline" && options.traffic_mode != "open-loop")
            {
                throw std::runtime_error("--traffic-mode must be offline or open-loop.");
            }
        }
        else if (arg == "--temperature")
        {
            options.temperature = parse_float(require_value("--temperature"), "temperature");
        }
        else if (arg == "--top-p")
        {
            options.top_p = parse_float(require_value("--top-p"), "top-p");
        }
        else if (arg == "--top-k")
        {
            options.top_k = parse_non_negative_int(require_value("--top-k"), "top-k");
        }
        else if (arg == "--repetition-penalty")
        {
            options.repetition_penalty = parse_float(require_value("--repetition-penalty"), "repetition-penalty");
            options.repetition_penalty_explicit = true;
        }
        else if (arg == "--seed")
        {
            options.seed = parse_uint64(require_value("--seed"), "seed");
        }
        else if (arg == "--stop-token-id")
        {
            options.stop_token_ids.push_back(parse_non_negative_int(require_value("--stop-token-id"), "stop-token-id"));
        }
        else if (arg == "--json")
        {
            options.json = true;
        }
        else if (arg == "--profile-detail")
        {
            options.profile_detail = true;
        }
        else if (arg == "--ignore-eos")
        {
            options.ignore_eos = true;
            options.ignore_eos_explicit = true;
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
    load_workload_jsonl(options);
    if (options.prompts.empty())
    {
        options.prompts = {"hello", "tiny llm inference"};
    }
    if (!options.request_ids.empty() && options.request_ids.size() != options.prompts.size())
    {
        throw std::runtime_error("request_ids and prompts must have the same length.");
    }
    if (options.request_ids.empty())
    {
        options.request_ids.reserve(options.prompts.size());
        for (size_t i = 0; i < options.prompts.size(); ++i)
        {
            options.request_ids.push_back("request-" + std::to_string(i));
        }
    }
    if (options.arrival_ms.empty())
    {
        options.arrival_ms.assign(options.prompts.size(), 0.0);
    }
    if (options.arrival_ms.size() != options.prompts.size())
    {
        throw std::runtime_error("arrival_ms and prompts must have the same length.");
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

CudaMemoryMetrics current_cuda_memory(const tiny_llm::ParallelConfig &parallel_config)
{
    CudaMemoryMetrics metrics;
#if TINYLLM_ENABLE_CUDA
    if (!parallel_config.is_cuda())
    {
        return metrics;
    }
    const auto stats =
        c10::cuda::CUDACachingAllocator::getDeviceStats(static_cast<c10::DeviceIndex>(parallel_config.device_id()));
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

void reset_cuda_peak_memory(const tiny_llm::ParallelConfig &parallel_config)
{
#if TINYLLM_ENABLE_CUDA
    if (parallel_config.is_cuda())
    {
        c10::cuda::CUDACachingAllocator::resetPeakStats(static_cast<c10::DeviceIndex>(parallel_config.device_id()));
    }
#else
    (void)parallel_config;
#endif
}

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool env_flag_enabled(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr)
    {
        return false;
    }
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

double mean(const std::vector<double> &values)
{
    if (values.empty())
    {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

PromptTokenStats count_prompt_tokens(const std::filesystem::path &model_dir, const std::vector<std::string> &prompts)
{
    tiny_llm::HFLlamaTokenizer tokenizer = tiny_llm::HFLlamaTokenizer::from_model_dir(model_dir.string());
    PromptTokenStats stats;
    stats.per_prompt.reserve(prompts.size());
    for (const std::string &prompt : prompts)
    {
        const int64_t count = static_cast<int64_t>(tokenizer.encode(prompt).size());
        stats.total += count;
        stats.per_prompt.push_back(count);
    }
    return stats;
}

size_t estimate_kv_num_blocks(const std::vector<int64_t> &prompt_token_counts, int32_t max_new_tokens,
                              int32_t block_size_tokens, int32_t num_layers)
{
    if (block_size_tokens <= 0 || num_layers <= 0)
    {
        throw std::runtime_error("invalid dimensions for KV block estimate.");
    }
    size_t required = 0;
    for (int64_t prompt_tokens : prompt_token_counts)
    {
        const int64_t total_tokens = prompt_tokens + max_new_tokens;
        const int64_t blocks_per_layer = (total_tokens + block_size_tokens - 1) / block_size_tokens;
        required += static_cast<size_t>(blocks_per_layer) * static_cast<size_t>(num_layers);
    }
    const size_t with_slack = (required * 6 + 4) / 5;
    return std::max<size_t>(256, with_slack);
}

RepeatMetrics run_once(const Options &options, tiny_llm::LLM &llm, double load_ms, bool measure)
{
    RepeatMetrics metrics;
    const bool open_loop = options.traffic_mode == "open-loop";
    metrics.request_traces.reserve(options.prompts.size());
    for (size_t i = 0; i < options.prompts.size(); ++i)
    {
        RequestTrace trace;
        trace.request_id = options.request_ids[i];
        trace.prompt_index = i;
        trace.submit_ms = open_loop && measure ? options.arrival_ms[i] : 0.0;
        trace.admit_ms = open_loop ? -1.0 : 0.0;
        if (i < options.prompt_token_counts.size())
        {
            trace.prompt_tokens = options.prompt_token_counts[i];
        }
        metrics.request_traces.push_back(std::move(trace));
    }

    tiny_llm::UserSamplingParams sampling_params;
    sampling_params.temperature = options.temperature;
    sampling_params.top_p = options.top_p;
    sampling_params.top_k = options.top_k;
    sampling_params.repetition_penalty = options.repetition_penalty;
    sampling_params.seed = options.seed;
    sampling_params.max_tokens = options.max_new_tokens;
    sampling_params.ignore_eos = options.ignore_eos;
    sampling_params.stop_token_ids = options.stop_token_ids;

    bool saw_first_token = false;
    Clock::time_point first_token_time{};
    if (measure)
    {
        reset_cuda_peak_memory(options.parallel_config);
    }
    const auto generation_start = Clock::now();
    std::vector<tiny_llm::CompletionOutput> outputs(options.prompts.size());
    tiny_llm::RuntimeProfilingStats profile;
    for (size_t i = 0; i < options.prompts.size(); ++i)
    {
        outputs[i].prompt = options.prompts[i];
    }

    auto record_output = [&](size_t prompt_index, const std::string &delta_text, int32_t token_id,
                             const std::string &text, const std::vector<int32_t> &token_ids, bool finished,
                             const std::string &finish_reason)
    {
        tiny_llm::CompletionOutput &completion = outputs[prompt_index];
        completion.text = text;
        completion.token_ids = token_ids;
        completion.finished = finished;
        completion.finish_reason = finish_reason;
        if (token_id >= 0 && !saw_first_token)
        {
            saw_first_token = true;
            first_token_time = Clock::now();
        }
        if (!measure || prompt_index >= metrics.request_traces.size())
        {
            return;
        }

        RequestTrace &trace = metrics.request_traces[prompt_index];
        const double event_ms = elapsed_ms(generation_start, Clock::now());
        if (token_id >= 0 && trace.first_token_ms < 0.0)
        {
            trace.first_token_ms = event_ms;
        }
        if (token_id >= 0)
        {
            TokenTraceEvent event;
            event.token_index = token_ids.empty() ? 0 : static_cast<int32_t>(token_ids.size() - 1);
            event.token_id = token_id;
            event.time_ms = event_ms;
            event.delta_text = delta_text;
            event.finished = finished;
            trace.tokens.push_back(std::move(event));
        }
        if (finished && trace.finish_ms < 0.0)
        {
            trace.finish_ms = event_ms;
            trace.finish_reason = finish_reason;
        }
    };

    if (!open_loop)
    {
        outputs = llm.generate(options.prompts, sampling_params,
                               [&](const tiny_llm::CompletionStreamOutput &output)
                               {
                                   record_output(output.prompt_index, output.delta_text, output.token_id, output.text,
                                                 output.token_ids, output.finished, output.finish_reason);
                               });
        profile = llm.last_generation_profile();
    }
    else
    {
        size_t next_request = 0;
        std::unordered_map<uint64_t, size_t> request_to_index;
        request_to_index.reserve(options.prompts.size());
        while (next_request < options.prompts.size() || llm.has_unfinished_requests())
        {
            if (!llm.has_unfinished_requests() && next_request < options.prompts.size())
            {
                const double target_ms = measure ? options.arrival_ms[next_request] : 0.0;
                const auto target_time = generation_start + std::chrono::duration_cast<Clock::duration>(
                                                                std::chrono::duration<double, std::milli>(target_ms));
                std::this_thread::sleep_until(target_time);
            }

            double now_ms = elapsed_ms(generation_start, Clock::now());
            while (next_request < options.prompts.size() &&
                   (measure ? options.arrival_ms[next_request] : 0.0) <= now_ms)
            {
                const uint64_t internal_id = llm.add_request(options.prompts[next_request], sampling_params);
                request_to_index.emplace(internal_id, next_request);
                if (measure)
                {
                    metrics.request_traces[next_request].admit_ms = elapsed_ms(generation_start, Clock::now());
                }
                ++next_request;
                now_ms = elapsed_ms(generation_start, Clock::now());
            }

            if (!llm.has_unfinished_requests())
            {
                continue;
            }

            const std::vector<tiny_llm::LLMStepOutput> step_outputs = llm.step();
            profile.add(llm.last_step_profile());
            tiny_llm::benchmark::require_step_progress(llm.has_unfinished_requests(),
                                                       llm.last_step_profile().scheduled_tokens);
            for (const tiny_llm::LLMStepOutput &output : step_outputs)
            {
                const auto it = request_to_index.find(output.request_id);
                if (it == request_to_index.end())
                {
                    throw std::runtime_error("open-loop benchmark received output for an unknown request.");
                }
                record_output(it->second, output.delta_text, output.token_id, output.text, output.token_ids,
                              output.finished, output.finish_reason);
            }
        }
    }
    const auto generation_end = Clock::now();

    if (!measure)
    {
        return metrics;
    }

    metrics.load_ms = load_ms;
    metrics.total_ms = elapsed_ms(generation_start, generation_end);
    metrics.first_token_ms = saw_first_token ? elapsed_ms(generation_start, first_token_time) : 0.0;
    metrics.prepare_inputs_ms = profile.prepare_inputs_ms;
    metrics.model_ms_total = profile.model_ms_total;
    metrics.prefill_ms = profile.prefill_ms;
    metrics.decode_ms_total = profile.decode_ms_total;
    metrics.mixed_model_ms = profile.mixed_model_ms;
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
    metrics.scheduled_requests = profile.scheduled_requests;
    metrics.scheduled_tokens = profile.scheduled_tokens;
    metrics.prefill_requests = profile.prefill_requests;
    metrics.decode_requests = profile.decode_requests;
    metrics.max_context_len = profile.max_context_len;
    metrics.profiled_steps = profile.profiled_steps;
    metrics.prompt_tokens = options.prompt_tokens;
    metrics.generated_tokens = 0;
    metrics.samples.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        const tiny_llm::CompletionOutput &output = outputs[i];
        metrics.generated_tokens += static_cast<int64_t>(output.token_ids.size());
        if (i < metrics.request_traces.size())
        {
            RequestTrace &trace = metrics.request_traces[i];
            trace.generated_tokens = static_cast<int64_t>(output.token_ids.size());
            if (trace.finish_ms < 0.0)
            {
                trace.finish_ms = metrics.total_ms;
                trace.finish_reason = output.finish_reason;
            }
        }
        SampleOutput sample;
        sample.request_id = i < options.request_ids.size() ? options.request_ids[i] : "request-" + std::to_string(i);
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

std::string json_escape(const std::string &text)
{
    std::ostringstream out;
    for (unsigned char ch : text)
    {
        switch (ch)
        {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (ch < 0x20)
            {
                const char *hex = "0123456789abcdef";
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

void print_json_int_array(const std::vector<int32_t> &values)
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

void print_samples_json(const std::vector<SampleOutput> &samples)
{
    std::cout << "[";
    for (size_t i = 0; i < samples.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ",";
        }
        const SampleOutput &sample = samples[i];
        std::cout << "{";
        std::cout << "\"request_id\":\"" << json_escape(sample.request_id) << "\",";
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

void print_request_metrics_json(const std::vector<RequestTrace> &traces)
{
    std::cout << "[";
    for (size_t i = 0; i < traces.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ",";
        }
        const RequestTrace &trace = traces[i];
        const double queue_ms = std::max(0.0, trace.admit_ms - trace.submit_ms);
        const double ttft_ms =
            trace.first_token_ms >= 0.0 ? std::max(0.0, trace.first_token_ms - trace.submit_ms) : 0.0;
        const double engine_ttft_ms =
            trace.first_token_ms >= 0.0 ? std::max(0.0, trace.first_token_ms - trace.admit_ms) : 0.0;
        const double tpot_ms = trace.generated_tokens > 1 ? std::max(0.0, trace.finish_ms - trace.first_token_ms) /
                                                                static_cast<double>(trace.generated_tokens - 1)
                                                          : 0.0;
        const double e2e_ms = trace.finish_ms >= 0.0 ? std::max(0.0, trace.finish_ms - trace.submit_ms) : 0.0;
        std::cout << "{";
        std::cout << "\"request_id\":\"" << json_escape(trace.request_id) << "\",";
        std::cout << "\"prompt_index\":" << trace.prompt_index << ",";
        std::cout << "\"prompt_tokens\":" << trace.prompt_tokens << ",";
        std::cout << "\"generated_tokens\":" << trace.generated_tokens << ",";
        std::cout << "\"submit_ms\":" << trace.submit_ms << ",";
        std::cout << "\"admit_ms\":" << trace.admit_ms << ",";
        std::cout << "\"first_token_ms\":" << trace.first_token_ms << ",";
        std::cout << "\"finish_ms\":" << trace.finish_ms << ",";
        std::cout << "\"queue_ms\":" << queue_ms << ",";
        std::cout << "\"ttft_ms\":" << ttft_ms << ",";
        std::cout << "\"engine_ttft_ms\":" << engine_ttft_ms << ",";
        std::cout << "\"tpot_ms\":" << tpot_ms << ",";
        std::cout << "\"e2e_ms\":" << e2e_ms << ",";
        std::cout << "\"finish_reason\":\"" << json_escape(trace.finish_reason) << "\"";
        std::cout << "}";
    }
    std::cout << "]";
}

void write_trace_events(const std::filesystem::path &path, int32_t repeat_index, const RepeatMetrics &metrics)
{
    if (path.empty())
    {
        return;
    }
    std::ofstream out(path, std::ios::app);
    if (!out)
    {
        throw std::runtime_error("failed to open events JSONL: " + path.string());
    }
    for (const RequestTrace &trace : metrics.request_traces)
    {
        out << "{\"repeat\":" << repeat_index << ",\"request_id\":\"" << json_escape(trace.request_id)
            << "\",\"prompt_index\":" << trace.prompt_index << ",\"event\":\"submit\",\"time_ms\":" << trace.submit_ms
            << "}\n";
        out << "{\"repeat\":" << repeat_index << ",\"request_id\":\"" << json_escape(trace.request_id)
            << "\",\"prompt_index\":" << trace.prompt_index << ",\"event\":\"admit\",\"time_ms\":" << trace.admit_ms
            << "}\n";
        for (const TokenTraceEvent &event : trace.tokens)
        {
            out << "{\"repeat\":" << repeat_index << ",\"request_id\":\"" << json_escape(trace.request_id)
                << "\",\"prompt_index\":" << trace.prompt_index << ",\"event\":\"token\",\"time_ms\":" << event.time_ms
                << ",\"token_index\":" << event.token_index << ",\"token_id\":" << event.token_id
                << ",\"finished\":" << (event.finished ? "true" : "false") << ",\"delta_text\":\""
                << json_escape(event.delta_text) << "\"}\n";
        }
        out << "{\"repeat\":" << repeat_index << ",\"request_id\":\"" << json_escape(trace.request_id)
            << "\",\"prompt_index\":" << trace.prompt_index << ",\"event\":\"finish\",\"time_ms\":" << trace.finish_ms
            << ",\"generated_tokens\":" << trace.generated_tokens << ",\"finish_reason\":\""
            << json_escape(trace.finish_reason) << "\"}\n";
    }
}

void print_summary(const Options &options, const std::vector<RepeatMetrics> &repeats)
{
    std::vector<double> load_ms;
    std::vector<double> total_ms;
    std::vector<double> first_token_ms;
    std::vector<double> prepare_inputs_ms;
    std::vector<double> model_ms_total;
    std::vector<double> prefill_ms;
    std::vector<double> decode_ms_total;
    std::vector<double> mixed_model_ms;
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
    std::vector<double> scheduled_requests;
    std::vector<double> scheduled_tokens;
    std::vector<double> prefill_requests;
    std::vector<double> decode_requests;
    std::vector<double> max_context_len;
    std::vector<double> profiled_steps;
    load_ms.reserve(repeats.size());
    total_ms.reserve(repeats.size());
    first_token_ms.reserve(repeats.size());
    prepare_inputs_ms.reserve(repeats.size());
    model_ms_total.reserve(repeats.size());
    prefill_ms.reserve(repeats.size());
    decode_ms_total.reserve(repeats.size());
    mixed_model_ms.reserve(repeats.size());
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
    scheduled_requests.reserve(repeats.size());
    scheduled_tokens.reserve(repeats.size());
    prefill_requests.reserve(repeats.size());
    decode_requests.reserve(repeats.size());
    max_context_len.reserve(repeats.size());
    profiled_steps.reserve(repeats.size());
    int64_t generated_tokens = 0;
    int64_t prompt_tokens = -1;
    int64_t prefill_tokens = 0;
    int64_t decode_tokens = 0;
    for (const RepeatMetrics &metrics : repeats)
    {
        load_ms.push_back(metrics.load_ms);
        total_ms.push_back(metrics.total_ms);
        first_token_ms.push_back(metrics.first_token_ms);
        prepare_inputs_ms.push_back(metrics.prepare_inputs_ms);
        model_ms_total.push_back(metrics.model_ms_total);
        prefill_ms.push_back(metrics.prefill_ms);
        decode_ms_total.push_back(metrics.decode_ms_total);
        mixed_model_ms.push_back(metrics.mixed_model_ms);
        decode_ms_per_token.push_back(metrics.decode_tokens > 0 && metrics.mixed_model_ms == 0.0
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
        scheduled_requests.push_back(static_cast<double>(metrics.scheduled_requests));
        scheduled_tokens.push_back(static_cast<double>(metrics.scheduled_tokens));
        prefill_requests.push_back(static_cast<double>(metrics.prefill_requests));
        decode_requests.push_back(static_cast<double>(metrics.decode_requests));
        max_context_len.push_back(static_cast<double>(metrics.max_context_len));
        profiled_steps.push_back(static_cast<double>(metrics.profiled_steps));
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
    const double avg_model_ms_total = mean(model_ms_total);
    const double avg_prefill_ms = mean(prefill_ms);
    const double avg_decode_ms_total = mean(decode_ms_total);
    const double avg_mixed_model_ms = mean(mixed_model_ms);
    const double avg_decode_ms_per_token = mean(decode_ms_per_token);
    const bool decode_ms_per_token_valid =
        std::all_of(mixed_model_ms.begin(), mixed_model_ms.end(), [](double value) { return value == 0.0; });
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
    const double avg_scheduled_requests = mean(scheduled_requests);
    const double avg_scheduled_tokens = mean(scheduled_tokens);
    const double avg_prefill_requests = mean(prefill_requests);
    const double avg_decode_requests = mean(decode_requests);
    const double avg_max_context_len = mean(max_context_len);
    const double avg_profiled_steps = mean(profiled_steps);
    const double avg_scheduled_requests_per_step =
        avg_profiled_steps > 0.0 ? avg_scheduled_requests / avg_profiled_steps : 0.0;
    const double avg_scheduled_tokens_per_step =
        avg_profiled_steps > 0.0 ? avg_scheduled_tokens / avg_profiled_steps : 0.0;
    const bool has_cuda_memory = !cuda_memory_allocated_mb.empty();
    const double avg_generated_tokens = repeats.empty() ? 0.0 : static_cast<double>(generated_tokens) / repeats.size();
    const double e2e_tokens_per_s = avg_total_ms > 0.0 ? avg_generated_tokens / (avg_total_ms / 1000.0) : 0.0;
    const double decode_ms = std::max(0.0, avg_total_ms - avg_first_ms);
    const double generated_decode_tokens =
        std::max(0.0, avg_generated_tokens - static_cast<double>(options.prompts.size()));
    const double decode_tokens_per_s = decode_ms > 0.0 ? generated_decode_tokens / (decode_ms / 1000.0) : 0.0;

    std::cout << "llama_engine_benchmark\n";
    std::cout << "  model: " << options.model_dir.string() << "\n";
    std::cout << "  device: " << options.device_text << "\n";
    std::cout << "  compute dtype: " << tiny_llm::runtime_dtype_name(options.compute_dtype) << "\n";
    std::cout << "  KV cache dtype: " << tiny_llm::runtime_dtype_name(options.kv_cache_dtype) << "\n";
    std::cout << "  traffic mode: " << options.traffic_mode << "\n";
    std::cout << "  prompts: " << options.prompts.size() << ", warmup: " << options.warmup
              << ", repeat: " << options.repeat << ", max_new_tokens: " << options.max_new_tokens
              << ", ignore_eos: " << (options.ignore_eos ? "on" : "off")
              << ", profile_detail: " << (options.profile_detail ? "on" : "off")
              << ", kv_num_blocks: " << options.kv_num_blocks << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  avg_load_init_ms: " << avg_load_ms << "\n";
    std::cout << "  avg_total_latency_ms: " << avg_total_ms << "\n";
    std::cout << "  avg_first_token_latency_ms: " << avg_first_ms << "\n";
    std::cout << "  avg_generated_tokens: " << avg_generated_tokens << "\n";
    std::cout << "  total_generated_tokens: " << generated_tokens << "\n";
    std::cout << "  prepare_inputs_ms: " << avg_prepare_inputs_ms << "\n";
    std::cout << "  model_ms_total: " << avg_model_ms_total << "\n";
    std::cout << "  prefill_ms: " << avg_prefill_ms << "\n";
    std::cout << "  decode_ms_total: " << avg_decode_ms_total << "\n";
    std::cout << "  mixed_model_ms: " << avg_mixed_model_ms << "\n";
    if (decode_ms_per_token_valid)
    {
        std::cout << "  decode_ms_per_token: " << avg_decode_ms_per_token << "\n";
    }
    else
    {
        std::cout << "  decode_ms_per_token: unavailable (mixed model steps)\n";
    }
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
    std::cout << "  avg_prefill_tokens: "
              << (repeats.empty() ? 0.0 : static_cast<double>(prefill_tokens) / repeats.size()) << "\n";
    std::cout << "  avg_decode_tokens: "
              << (repeats.empty() ? 0.0 : static_cast<double>(decode_tokens) / repeats.size()) << "\n";
    std::cout << "  avg_scheduled_requests: " << avg_scheduled_requests << "\n";
    std::cout << "  avg_scheduled_tokens: " << avg_scheduled_tokens << "\n";
    std::cout << "  avg_profiled_steps: " << avg_profiled_steps << "\n";
    std::cout << "  avg_scheduled_requests_per_step: " << avg_scheduled_requests_per_step << "\n";
    std::cout << "  avg_scheduled_tokens_per_step: " << avg_scheduled_tokens_per_step << "\n";
    std::cout << "  avg_prefill_requests: " << avg_prefill_requests << "\n";
    std::cout << "  avg_decode_requests: " << avg_decode_requests << "\n";
    std::cout << "  avg_max_context_len: " << avg_max_context_len << "\n";
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
            const SampleOutput &sample = repeats.front().samples[i];
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
    std::cout << "\"compute_dtype\":\"" << tiny_llm::runtime_dtype_name(options.compute_dtype) << "\",";
    std::cout << "\"kv_cache_dtype\":\"" << tiny_llm::runtime_dtype_name(options.kv_cache_dtype) << "\",";
    std::cout << "\"traffic_mode\":\"" << options.traffic_mode << "\",";
    std::cout << "\"prompt_count\":" << options.prompts.size() << ",";
    std::cout << "\"prompt_tokens\":" << prompt_tokens << ",";
    std::cout << "\"warmup\":" << options.warmup << ",";
    std::cout << "\"repeat\":" << options.repeat << ",";
    std::cout << "\"max_new_tokens\":" << options.max_new_tokens << ",";
    std::cout << "\"temperature\":" << options.temperature << ",";
    std::cout << "\"top_p\":" << options.top_p << ",";
    std::cout << "\"top_k\":" << options.top_k << ",";
    std::cout << "\"repetition_penalty\":" << options.repetition_penalty << ",";
    std::cout << "\"seed\":" << options.seed << ",";
    std::cout << "\"ignore_eos\":" << (options.ignore_eos ? "true" : "false") << ",";
    std::cout << "\"profile_detail\":" << (options.profile_detail ? "true" : "false") << ",";
    std::cout << "\"workload_jsonl\":\"" << json_escape(options.workload_jsonl.string()) << "\",";
    std::cout << "\"events_jsonl\":\"" << json_escape(options.events_jsonl.string()) << "\",";
    std::cout << "\"kv_num_blocks\":" << options.kv_num_blocks << ",";
    std::cout << "\"max_num_batched_tokens\":" << options.max_num_batched_tokens << ",";
    std::cout << "\"avg_load_init_ms\":" << avg_load_ms << ",";
    std::cout << "\"avg_total_latency_ms\":" << avg_total_ms << ",";
    std::cout << "\"avg_first_token_latency_ms\":" << avg_first_ms << ",";
    std::cout << "\"avg_generated_tokens\":" << avg_generated_tokens << ",";
    std::cout << "\"total_generated_tokens\":" << generated_tokens << ",";
    std::cout << "\"prepare_inputs_ms\":" << avg_prepare_inputs_ms << ",";
    std::cout << "\"model_ms_total\":" << avg_model_ms_total << ",";
    std::cout << "\"prefill_ms\":" << avg_prefill_ms << ",";
    std::cout << "\"decode_ms_total\":" << avg_decode_ms_total << ",";
    std::cout << "\"mixed_model_ms\":" << avg_mixed_model_ms << ",";
    std::cout << "\"decode_ms_per_token\":" << avg_decode_ms_per_token << ",";
    std::cout << "\"decode_ms_per_token_valid\":" << (decode_ms_per_token_valid ? "true" : "false") << ",";
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
    std::cout << "\"avg_prefill_tokens\":"
              << (repeats.empty() ? 0.0 : static_cast<double>(prefill_tokens) / repeats.size()) << ",";
    std::cout << "\"avg_decode_tokens\":"
              << (repeats.empty() ? 0.0 : static_cast<double>(decode_tokens) / repeats.size()) << ",";
    std::cout << "\"avg_scheduled_requests\":" << avg_scheduled_requests << ",";
    std::cout << "\"avg_scheduled_tokens\":" << avg_scheduled_tokens << ",";
    std::cout << "\"avg_profiled_steps\":" << avg_profiled_steps << ",";
    std::cout << "\"avg_scheduled_requests_per_step\":" << avg_scheduled_requests_per_step << ",";
    std::cout << "\"avg_scheduled_tokens_per_step\":" << avg_scheduled_tokens_per_step << ",";
    std::cout << "\"avg_prefill_requests\":" << avg_prefill_requests << ",";
    std::cout << "\"avg_decode_requests\":" << avg_decode_requests << ",";
    std::cout << "\"avg_max_context_len\":" << avg_max_context_len << ",";
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
    std::cout << "\"request_metrics\":";
    if (!repeats.empty())
    {
        print_request_metrics_json(repeats.front().request_traces);
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

int main(int argc, char **argv)
{
    try
    {
        Options options = parse_args(argc, argv);
        options.profile_detail = options.profile_detail || env_flag_enabled("TINYLLM_PROFILE_DETAIL");
        if (options.profile_detail)
        {
            setenv("TINYLLM_PROFILE_DETAIL", "1", 1);
        }
        if (!options.repetition_penalty_explicit)
        {
            const tiny_llm::GenerationConfig generation_config =
                tiny_llm::load_generation_config_from_dir(options.model_dir.string());
            options.repetition_penalty = generation_config.repetition_penalty;
        }
        const PromptTokenStats token_stats = count_prompt_tokens(options.model_dir, options.prompts);
        options.prompt_tokens = token_stats.total;
        options.prompt_token_counts = token_stats.per_prompt;
        const tiny_llm::LlamaConfig hf_config =
            tiny_llm::HFLlamaConfigLoader::load_from_dir(options.model_dir.string());
        if (!options.max_num_batched_tokens_explicit)
        {
            options.max_num_batched_tokens =
                std::max<int32_t>(1, std::min<int64_t>(static_cast<int64_t>(options.max_num_batched_token_cap),
                                                       std::max<int64_t>(1, options.prompt_tokens)));
        }
        constexpr int32_t kBlockSizeTokens = 16;
        if (!options.kv_num_blocks_explicit)
        {
            options.kv_num_blocks = estimate_kv_num_blocks(options.prompt_token_counts, options.max_new_tokens,
                                                           kBlockSizeTokens, hf_config.num_hidden_layers);
        }

        const auto load_start = Clock::now();
        tiny_llm::LLMOptions llm_options(options.model_dir.string(), options.parallel_config);
        llm_options.compute_dtype = options.compute_dtype;
        llm_options.kv_cache_dtype = options.kv_cache_dtype;
        llm_options.max_tokens = options.max_new_tokens;
        llm_options.scheduler_config.max_prefill_tokens_per_step = options.max_num_batched_tokens;
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
        if (!options.events_jsonl.empty())
        {
            if (options.events_jsonl.has_parent_path())
            {
                std::filesystem::create_directories(options.events_jsonl.parent_path());
            }
            std::ofstream clear_events(options.events_jsonl, std::ios::trunc);
            if (!clear_events)
            {
                throw std::runtime_error("failed to create events JSONL: " + options.events_jsonl.string());
            }
        }
        for (int32_t i = 0; i < options.repeat; ++i)
        {
            RepeatMetrics metrics = run_once(options, llm, load_ms, true);
            write_trace_events(options.events_jsonl, i, metrics);
            repeats.push_back(std::move(metrics));
        }
        print_summary(options, repeats);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "llama_engine_benchmark failed: " << ex.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
