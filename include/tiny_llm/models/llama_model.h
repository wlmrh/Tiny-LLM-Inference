#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/models/llama_config.h"
#include "tiny_llm/models/llama_decoder_layer.h"
#include "tiny_llm/models/llama_weight_map.h"
#include "tiny_llm/models/model.h"
#include "tiny_llm/models/modules/embedding.h"
#include "tiny_llm/models/modules/linear.h"
#include "tiny_llm/models/modules/rmsnorm.h"
#include "tiny_llm/runtime/parallel_config.h"

namespace tiny_llm {

struct LlamaModelBuffers {
    Tensor hidden_states;
    Tensor residual;
    Tensor norm_output;
    LlamaDecoderLayerBuffers layer;
};

class LlamaModel : public torch::nn::Module {
public:
    LlamaModel(LlamaConfig config, const WeightMap& weight_map);

    void allocate_buffers(int max_batch_size);
    void allocate_buffers(int max_batch_size, const ParallelConfig& parallel_config);

    int32_t num_layers() const { return config_.num_hidden_layers; }
    int32_t vocab_size() const { return config_.vocab_size; }
    int32_t hidden_size() const { return config_.hidden_size; }
    const LlamaConfig& config() const { return config_; }

    Tensor forward_hidden(const PreparedInputs& inputs, RuntimeContext& ctx);

private:
    void validate_forward_inputs(const Tensor& input_ids,
                                 const Tensor& positions) const;
    void validate_weight_shapes(const WeightMap& weight_map) const;
    LlamaModelBuffers make_batch_buffers(int batch_size) const;
    Tensor make_batch_view_2d(const Tensor& backing, int batch_size, int width) const;

    LlamaConfig config_;
    std::vector<std::shared_ptr<LlamaDecoderLayer>> layers_;
    std::shared_ptr<modules::Embedding> embed_tokens_;
    std::shared_ptr<modules::RMSNorm> final_norm_;
    int32_t allocated_max_batch_size_ = 0;
    ParallelConfig buffer_parallel_config_{};
    LlamaModelBuffers buffers_;
};

class LlamaForCausalLM : public Model {
public:
    LlamaForCausalLM(LlamaConfig config, WeightMap weight_map);

    void allocate_buffers(int max_batch_size);
    void allocate_buffers(int max_batch_size, const ParallelConfig& parallel_config);

    int32_t num_layers() const override { return model_->num_layers(); }
    int32_t vocab_size() const override { return config_.vocab_size; }
    int32_t expected_bos_id() const override { return config_.bos_token_id; }
    int32_t expected_eos_id() const override { return config_.eos_token_id; }
    int32_t expected_unk_id() const override { return config_.unk_token_id; }

    Tensor forward(const PreparedInputs& inputs, RuntimeContext& ctx) override;
    Tensor compute_logits(const Tensor& hidden_states, RuntimeContext& ctx) const;

private:
    void bind_lm_head(const WeightMap& weight_map);

    LlamaConfig config_;
    std::shared_ptr<LlamaModel> model_;
    std::shared_ptr<modules::Linear> lm_head_;
};

} // namespace tiny_llm
