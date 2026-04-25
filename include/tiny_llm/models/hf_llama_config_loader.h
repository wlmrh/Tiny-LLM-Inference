#pragma once

#include <string>

#include "tiny_llm/models/llama_config.h"

namespace tiny_llm {

class HFLlamaConfigLoader {
public:
    static LlamaConfig load_from_dir(const std::string& hf_model_dir);
    static LlamaConfig load_from_files(const std::string& config_file_path,
                                       const std::string& tokenizer_config_file_path = "");
};

} // namespace tiny_llm
