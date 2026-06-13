#pragma once

#include <string>

namespace tiny_llm {

struct GenerationConfig {
    float repetition_penalty = 1.0f;
};

GenerationConfig load_generation_config_from_dir(const std::string& model_dir);

} // namespace tiny_llm
