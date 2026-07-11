#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tiny_llm
{

/**
 * @brief Runtime tokenizer contract for request encoding and result decoding.
 */
class Tokenizer
{
  public:
    virtual ~Tokenizer() = default;

    virtual std::vector<int32_t> encode(const std::string &text) const = 0;
    virtual std::string decode(const std::vector<int32_t> &ids) const = 0;

    virtual int32_t vocab_size() const = 0;
    virtual int32_t bos_id() const = 0;
    virtual int32_t eos_id() const = 0;
    virtual int32_t unk_id() const = 0;

    /**
     * @brief Returns true when id is valid for this tokenizer.
     */
    virtual bool is_valid_token_id(int32_t id) const = 0;
};

/**
 * @brief Lightweight HuggingFace LLaMA tokenizer from tokenizer.json.
 */
class HFLlamaTokenizer final : public Tokenizer
{
  public:
    static HFLlamaTokenizer from_model_dir(const std::string &hf_model_dir);
    ~HFLlamaTokenizer() override;

    HFLlamaTokenizer(HFLlamaTokenizer &&) noexcept;
    HFLlamaTokenizer &operator=(HFLlamaTokenizer &&) noexcept;
    HFLlamaTokenizer(const HFLlamaTokenizer &) = delete;
    HFLlamaTokenizer &operator=(const HFLlamaTokenizer &) = delete;

    std::vector<int32_t> encode(const std::string &text) const override;
    std::string decode(const std::vector<int32_t> &ids) const override;

    int32_t vocab_size() const override;
    int32_t bos_id() const override;
    int32_t eos_id() const override;
    int32_t unk_id() const override;
    bool is_valid_token_id(int32_t id) const override;

  private:
    struct Impl;
    explicit HFLlamaTokenizer(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t unk_id_ = -1;
};

} // namespace tiny_llm
