#pragma once

#include <cstdint>
#include <string>

namespace tiny_llm
{

/**
 * @brief Configuration for HuggingFace-style Llama-family checkpoints.
 *
 * Field names intentionally follow `config.json` so one config struct can
 * describe Llama-2, Llama-3 and smaller compatible variants such as SmolLM.
 */
struct LlamaConfig
{
    int32_t vocab_size = 32000;
    int32_t hidden_size = 512;
    int32_t intermediate_size = 2048;
    int32_t num_hidden_layers = 4;
    int32_t num_attention_heads = 8;
    int32_t num_key_value_heads = 8;
    int32_t head_dim = 64;
    int32_t max_position_embeddings = 2048;
    int32_t bos_token_id = -1;
    int32_t eos_token_id = -1;
    int32_t unk_token_id = -1;
    int32_t pad_token_id = -1;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::string rope_scaling_type;
    float rope_scaling_factor = 1.0f;
    float rope_scaling_low_freq_factor = 1.0f;
    float rope_scaling_high_freq_factor = 1.0f;
    int32_t rope_scaling_original_max_position_embeddings = 0;
    std::string hidden_act = "silu";
    std::string model_type = "llama";
    std::string torch_dtype = "float32";
};

} // namespace tiny_llm
