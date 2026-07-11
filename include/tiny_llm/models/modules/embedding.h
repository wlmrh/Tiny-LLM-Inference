#pragma once

#include <cstdint>

#include "tiny_llm/core/tensor.h"

namespace tiny_llm
{
namespace modules
{

enum class EmbeddingLayout
{
    kVocabHidden = 0,
    kHiddenVocab = 1,
};

class Embedding : public torch::nn::Module
{
  public:
    Embedding(int32_t vocab_size, int32_t hidden_size);

    void bind_weight(const Tensor &weight);
    void forward(const Tensor &ids, Tensor &output) const;

    int32_t vocab_size() const
    {
        return vocab_size_;
    }
    int32_t hidden_size() const
    {
        return hidden_size_;
    }

  private:
    Tensor weight_;
    EmbeddingLayout layout_ = EmbeddingLayout::kVocabHidden;
    int32_t vocab_size_ = 0;
    int32_t hidden_size_ = 0;
};

} // namespace modules
} // namespace tiny_llm
