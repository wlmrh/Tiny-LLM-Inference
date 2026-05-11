#include "tiny_llm/models/modules/embedding.h"

#include "tiny_llm/operators/llama_ops.h"

#include <stdexcept>

namespace tiny_llm {
namespace modules {

Embedding::Embedding(int32_t vocab_size, int32_t hidden_size)
    : vocab_size_(vocab_size), hidden_size_(hidden_size)
{
    if (vocab_size_ <= 0 || hidden_size_ <= 0)
    {
        throw std::runtime_error("modules::Embedding: dimensions must be positive.");
    }
}

void Embedding::bind_weight(const Tensor& weight)
{
    if (!weight.defined() || tensor_dtype(weight) != DType::kFloat32 || weight.dim() != 2)
    {
        throw std::runtime_error("modules::Embedding::bind_weight: weight must be rank-2 float32.");
    }
    if (weight.size(0) == vocab_size_ && weight.size(1) == hidden_size_)
    {
        layout_ = EmbeddingLayout::kVocabHidden;
    }
    else if (weight.size(0) == hidden_size_ && weight.size(1) == vocab_size_)
    {
        layout_ = EmbeddingLayout::kHiddenVocab;
    }
    else
    {
        throw std::runtime_error("modules::Embedding::bind_weight: unsupported weight shape.");
    }
    weight_ = register_parameter("weight", weight, /*requires_grad=*/false);
}

void Embedding::forward(const Tensor& ids, Tensor& output) const
{
    if (!weight_.defined())
    {
        throw std::runtime_error("modules::Embedding::forward: weight is not bound.");
    }
    if (tensor_dtype(ids) != DType::kInt32 || tensor_dtype(output) != DType::kFloat32)
    {
        throw std::runtime_error("modules::Embedding::forward: dtype mismatch.");
    }
    if (ids.dim() != 1 || output.dim() != 2 || output.size(0) != ids.size(0) || output.size(1) != hidden_size_)
    {
        throw std::runtime_error("modules::Embedding::forward: shape mismatch.");
    }
    ops::embedding_lookup(
        ids,
        weight_,
        output,
        vocab_size_,
        hidden_size_,
        layout_ == EmbeddingLayout::kVocabHidden);
}

} // namespace modules
} // namespace tiny_llm
