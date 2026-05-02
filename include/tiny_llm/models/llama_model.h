#pragma once

#include <cstdint>
#include <vector>

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/models/llama_config.h"
#include "tiny_llm/models/llama_decoder_layer.h"
#include "tiny_llm/models/llama_weight_map.h"
#include "tiny_llm/models/model.h"
#include "tiny_llm/models/modules/linear.h"
#include "tiny_llm/models/modules/rmsnorm.h"
#include "tiny_llm/runtime/parallel_config.h"

namespace tiny_llm {

class ExecutionContext;

struct LlamaModelBuffers {
    Tensor hidden_states;
    Tensor residual;
    Tensor norm_output;
    LlamaDecoderLayerBuffers layer;
};

class LlamaModel : public Model {
public:
    LlamaModel(LlamaConfig config, WeightMap weight_map);

    void allocate_buffers(int max_batch_size);
    void allocate_buffers(int max_batch_size, const ParallelConfig& parallel_config);

    int32_t num_layers() const override { return config_.num_hidden_layers; }
    int32_t vocab_size() const override { return config_.vocab_size; }
    int32_t expected_bos_id() const override { return config_.bos_token_id; }
    int32_t expected_eos_id() const override { return config_.eos_token_id; }
    int32_t expected_unk_id() const override { return config_.unk_token_id; }

    void forward_step(const Tensor& input_ids,
                      const Tensor& positions,
                      Tensor& logits,
                      ExecutionContext& ctx) override;

private:
    enum class EmbeddingLayout {
        kVocabHidden = 0,
        kHiddenVocab = 1,
    };

    void validate_forward_inputs(const Tensor& input_ids,
                                 const Tensor& positions,
                                 const Tensor& logits) const;
    void validate_weight_shapes();
    void bind_top_level_weights();
    void lookup_embedding(const Tensor& ids, Tensor& out) const;
    LlamaModelBuffers make_batch_buffers(int batch_size) const;
    Tensor make_batch_view_2d(const Tensor& backing, int batch_size, int width) const;

    LlamaConfig config_;
    WeightMap weight_map_;
    std::vector<LlamaDecoderLayer> layers_;
    Tensor embed_tokens_;
    modules::RMSNorm final_norm_;
    modules::Linear lm_head_;
    EmbeddingLayout embedding_layout_ = EmbeddingLayout::kVocabHidden;
    int32_t allocated_max_batch_size_ = 0;
    ParallelConfig buffer_parallel_config_{};
    LlamaModelBuffers buffers_;
};

} // namespace tiny_llm
