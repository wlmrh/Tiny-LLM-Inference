#pragma once

#include <cstdint>

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/prepared_inputs.h"
#include "tiny_llm/runtime/runtime_context.h"

namespace tiny_llm {

/**
 * @brief Runtime model contract used by ModelRunner.
 */
class Model : public torch::nn::Module {
public:
    virtual ~Model() = default;

    /**
     * @brief Returns transformer layer count used by KV metadata.
     */
    virtual int32_t num_layers() const = 0;

    /**
     * @brief Returns vocabulary size expected by logits output.
     */
    virtual int32_t vocab_size() const = 0;

    /**
     * @brief Computes logits for a flattened scheduler batch.
     */
    virtual Tensor forward(const PreparedInputs& inputs, RuntimeContext& ctx) = 0;

    /**
     * @brief Expected BOS token id for model/tokenizer contract checks.
     * Returns -1 when unconstrained.
     */
    virtual int32_t expected_bos_id() const { return -1; }

    /**
     * @brief Expected EOS token id for model/tokenizer contract checks.
     * Returns -1 when unconstrained.
     */
    virtual int32_t expected_eos_id() const { return -1; }

    /**
     * @brief Expected UNK token id for model/tokenizer contract checks.
     * Returns -1 when unconstrained.
     */
    virtual int32_t expected_unk_id() const { return -1; }
};

} // namespace tiny_llm
