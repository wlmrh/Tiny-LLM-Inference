#pragma once

#include <cstdint>

namespace tiny_llm {

class ExecutionContext;
class Tensor;

/**
 * @brief Runtime model contract used by the scheduler.
 */
class Model {
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
     * @brief Performs one decode step and writes next-token logits.
     */
    virtual void forward_step(const Tensor& input_ids,
                              const Tensor& positions,
                              Tensor& logits,
                              ExecutionContext& ctx) = 0;

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