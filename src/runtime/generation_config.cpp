#include "tiny_llm/runtime/generation_config.h"

#include "../common/hf_json.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tiny_llm
{

namespace
{

std::string read_text_file(const std::string &path, const std::string &error_prefix)
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

float read_optional_float(const hf_json::Value &root, const std::string &key, float default_value,
                          const std::string &error_prefix)
{
    const hf_json::Value *value = hf_json::find_object_field(root, key, error_prefix);
    if (value == nullptr || value->type == hf_json::ValueType::kNull)
    {
        return default_value;
    }

    return static_cast<float>(value->as_number(error_prefix + ": " + key));
}

GenerationConfig load_generation_config_file(const std::string &path)
{
    const std::string error_prefix = "load_generation_config_file";
    const std::string content = read_text_file(path, error_prefix);
    const hf_json::Value root = hf_json::parse(content, error_prefix);

    GenerationConfig config;
    config.repetition_penalty = read_optional_float(root, "repetition_penalty", 1.0f, error_prefix);
    return config;
}

} // namespace

GenerationConfig load_generation_config_from_dir(const std::string &model_dir)
{
    const std::filesystem::path config_path = std::filesystem::path(model_dir) / "generation_config.json";
    if (!std::filesystem::exists(config_path))
    {
        return GenerationConfig{};
    }
    return load_generation_config_file(config_path.string());
}

} // namespace tiny_llm
