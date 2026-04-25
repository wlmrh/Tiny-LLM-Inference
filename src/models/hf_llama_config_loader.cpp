#include "tiny_llm/models/hf_llama_config_loader.h"

#include "hf_json.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tiny_llm {

namespace {

std::string read_text_file(const std::string& path, const std::string& error_prefix)
{
    std::ifstream fin(path);
    if (!fin)
    {
        throw std::runtime_error(error_prefix + ": failed to open file: " + path);
    }

    std::ostringstream ss;
    ss << fin.rdbuf();
    return ss.str();
}

hf_json::Value parse_json_file(const std::string& path, const std::string& error_prefix)
{
    const std::string content = read_text_file(path, error_prefix);
    return hf_json::parse(content, error_prefix);
}

int32_t checked_to_int32(int64_t value, const std::string& error_prefix)
{
    if (value < static_cast<int64_t>(std::numeric_limits<int32_t>::min())
        || value > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
    {
        throw std::runtime_error(error_prefix + ": value is out of int32 range.");
    }
    return static_cast<int32_t>(value);
}

int32_t read_required_int(const hf_json::Value& root,
                          const std::string& key,
                          const std::string& error_prefix)
{
    const hf_json::Value& value = hf_json::require_object_field(root, key, error_prefix);
    return checked_to_int32(value.as_int64(error_prefix + ": " + key), error_prefix + ": " + key);
}

std::optional<int32_t> read_optional_int(const hf_json::Value& root,
                                         const std::string& key,
                                         const std::string& error_prefix)
{
    const hf_json::Value* value = hf_json::find_object_field(root, key, error_prefix);
    if (value == nullptr)
    {
        return std::nullopt;
    }

    return checked_to_int32(value->as_int64(error_prefix + ": " + key), error_prefix + ": " + key);
}

float read_optional_float(const hf_json::Value& root,
                          const std::string& key,
                          float default_value,
                          const std::string& error_prefix)
{
    const hf_json::Value* value = hf_json::find_object_field(root, key, error_prefix);
    if (value == nullptr)
    {
        return default_value;
    }

    return static_cast<float>(value->as_number(error_prefix + ": " + key));
}

int32_t resolve_token_id(const std::string& token_name,
                         const std::optional<int32_t>& config_id,
                         const std::optional<int32_t>& tokenizer_id,
                         bool required,
                         const std::string& error_prefix)
{
    if (config_id.has_value() && tokenizer_id.has_value() && config_id.value() != tokenizer_id.value())
    {
        throw std::runtime_error(
            error_prefix + ": token id mismatch for " + token_name
            + " (config.json=" + std::to_string(config_id.value())
            + ", tokenizer_config.json=" + std::to_string(tokenizer_id.value()) + ").");
    }

    if (tokenizer_id.has_value())
    {
        return tokenizer_id.value();
    }

    if (config_id.has_value())
    {
        return config_id.value();
    }

    if (required)
    {
        throw std::runtime_error(error_prefix + ": missing required token id for " + token_name + ".");
    }

    return -1;
}

} // namespace

LlamaConfig HFLlamaConfigLoader::load_from_dir(const std::string& hf_model_dir)
{
    if (hf_model_dir.empty())
    {
        throw std::runtime_error("HFLlamaConfigLoader::load_from_dir: hf_model_dir must be non-empty.");
    }

    const std::filesystem::path model_dir(hf_model_dir);
    const std::filesystem::path config_path = model_dir / "config.json";
    const std::filesystem::path tokenizer_config_path = model_dir / "tokenizer_config.json";

    if (std::filesystem::exists(tokenizer_config_path))
    {
        return load_from_files(config_path.string(), tokenizer_config_path.string());
    }

    return load_from_files(config_path.string(), "");
}

LlamaConfig HFLlamaConfigLoader::load_from_files(const std::string& config_file_path,
                                                 const std::string& tokenizer_config_file_path)
{
    constexpr const char* kErr = "HFLlamaConfigLoader::load_from_files";

    const hf_json::Value config_root = parse_json_file(config_file_path, kErr);

    LlamaConfig config;
    if (const hf_json::Value* model_type = hf_json::find_object_field(config_root, "model_type", kErr))
    {
        config.model_type = model_type->as_string(std::string(kErr) + ": model_type");
    }
    config.hidden_size = read_required_int(config_root, "hidden_size", kErr);
    config.intermediate_size = read_required_int(config_root, "intermediate_size", kErr);
    config.num_hidden_layers = read_required_int(config_root, "num_hidden_layers", kErr);
    config.num_attention_heads = read_required_int(config_root, "num_attention_heads", kErr);
    config.num_key_value_heads =
        read_optional_int(config_root, "num_key_value_heads", kErr).value_or(config.num_attention_heads);
    config.max_position_embeddings =
        read_optional_int(config_root, "max_position_embeddings", kErr).value_or(config.max_position_embeddings);
    config.vocab_size = read_required_int(config_root, "vocab_size", kErr);
    config.rms_norm_eps = read_optional_float(config_root, "rms_norm_eps", 1e-6f, kErr);
    config.rope_theta = read_optional_float(config_root, "rope_theta", config.rope_theta, kErr);
    config.pad_token_id = read_optional_int(config_root, "pad_token_id", kErr).value_or(config.pad_token_id);
    if (const hf_json::Value* hidden_act = hf_json::find_object_field(config_root, "hidden_act", kErr))
    {
        config.hidden_act = hidden_act->as_string(std::string(kErr) + ": hidden_act");
    }
    if (const hf_json::Value* torch_dtype = hf_json::find_object_field(config_root, "torch_dtype", kErr))
    {
        config.torch_dtype = torch_dtype->as_string(std::string(kErr) + ": torch_dtype");
    }

    if (config.num_hidden_layers <= 0 || config.hidden_size <= 0 || config.intermediate_size <= 0
        || config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.vocab_size <= 0)
    {
        throw std::runtime_error(
            "HFLlamaConfigLoader::load_from_files: invalid non-positive model dimensions in config.json.");
    }

    if (config.hidden_size % config.num_attention_heads != 0)
    {
        throw std::runtime_error(
            "HFLlamaConfigLoader::load_from_files: hidden_size must be divisible by num_attention_heads.");
    }
    if (config.num_attention_heads % config.num_key_value_heads != 0)
    {
        throw std::runtime_error(
            "HFLlamaConfigLoader::load_from_files: num_attention_heads must be divisible by num_key_value_heads.");
    }
    config.head_dim = config.hidden_size / config.num_attention_heads;

    const std::optional<int32_t> config_bos = read_optional_int(config_root, "bos_token_id", kErr);
    const std::optional<int32_t> config_eos = read_optional_int(config_root, "eos_token_id", kErr);
    const std::optional<int32_t> config_unk = read_optional_int(config_root, "unk_token_id", kErr);

    std::optional<int32_t> tokenizer_bos;
    std::optional<int32_t> tokenizer_eos;
    std::optional<int32_t> tokenizer_unk;

    if (!tokenizer_config_file_path.empty())
    {
        if (!std::filesystem::exists(tokenizer_config_file_path))
        {
            throw std::runtime_error(
                "HFLlamaConfigLoader::load_from_files: tokenizer_config.json does not exist: "
                + tokenizer_config_file_path);
        }

        const hf_json::Value tokenizer_root = parse_json_file(tokenizer_config_file_path, kErr);
        tokenizer_bos = read_optional_int(tokenizer_root, "bos_token_id", kErr);
        tokenizer_eos = read_optional_int(tokenizer_root, "eos_token_id", kErr);
        tokenizer_unk = read_optional_int(tokenizer_root, "unk_token_id", kErr);
    }

    config.bos_token_id = resolve_token_id("bos_token_id", config_bos, tokenizer_bos, true, kErr);
    config.eos_token_id = resolve_token_id("eos_token_id", config_eos, tokenizer_eos, true, kErr);
    config.unk_token_id = resolve_token_id("unk_token_id", config_unk, tokenizer_unk, false, kErr);

    if (config.bos_token_id < 0 || config.bos_token_id >= config.vocab_size
        || config.eos_token_id < 0 || config.eos_token_id >= config.vocab_size)
    {
        throw std::runtime_error(
            "HFLlamaConfigLoader::load_from_files: bos/eos token id is out of vocab range.");
    }

    if (config.unk_token_id >= 0 && config.unk_token_id >= config.vocab_size)
    {
        throw std::runtime_error(
            "HFLlamaConfigLoader::load_from_files: unk token id is out of vocab range.");
    }

    return config;
}

} // namespace tiny_llm
