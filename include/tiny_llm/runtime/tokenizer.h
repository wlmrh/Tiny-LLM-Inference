#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tiny_llm {

/**
 * @brief Runtime tokenizer contract for request encoding and result decoding.
 */
class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<int32_t> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<int32_t>& ids) const = 0;

    virtual int32_t vocab_size() const = 0;
    virtual int32_t bos_id() const = 0;
    virtual int32_t eos_id() const = 0;
    virtual int32_t unk_id() const = 0;

    /**
     * @brief Returns true when tokenizer vocabulary cannot grow at runtime.
     */
    virtual bool is_fixed_vocab() const = 0;

    /**
     * @brief Returns true when id is valid for this tokenizer.
     */
    virtual bool is_valid_token_id(int32_t id) const = 0;
};

/**
 * @brief Lightweight tokenizer holder for dependency injection.
 */
class TokenizerRegistry {
public:
    explicit TokenizerRegistry(Tokenizer* tokenizer);

    Tokenizer* mutable_tokenizer() const { return tokenizer_; }
    const Tokenizer* tokenizer() const { return tokenizer_; }

private:
    Tokenizer* tokenizer_ = nullptr;
};

/**
 * @brief Minimal whitespace tokenizer for runtime demo and tests.
 */
class WhitespaceTokenizer final : public Tokenizer {
public:
    explicit WhitespaceTokenizer(int32_t max_vocab_size = 32000);

    std::vector<int32_t> encode(const std::string& text) override;
    std::string decode(const std::vector<int32_t>& ids) const override;

    int32_t vocab_size() const override;
    int32_t bos_id() const override;
    int32_t eos_id() const override;
    int32_t unk_id() const override;
    bool is_fixed_vocab() const override;
    bool is_valid_token_id(int32_t id) const override;

private:
    int32_t max_vocab_size_ = 0;
    int32_t pad_id_ = 0;
    int32_t bos_id_ = 1;
    int32_t eos_id_ = 2;
    int32_t unk_id_ = 3;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
};

/**
 * @brief WordPiece tokenizer backed by a BERT-style vocab.txt file.
 */
class WordPieceTokenizer final : public Tokenizer {
public:
    static WordPieceTokenizer from_vocab_file(const std::string& path,
                                              const std::string& pad_token = "[PAD]",
                                              const std::string& bos_token = "[BOS]",
                                              const std::string& eos_token = "[EOS]",
                                              const std::string& unk_token = "[UNK]",
                                              bool do_lower_case = true);

    std::vector<int32_t> encode(const std::string& text) override;
    std::string decode(const std::vector<int32_t>& ids) const override;

    int32_t vocab_size() const override;
    int32_t bos_id() const override;
    int32_t eos_id() const override;
    int32_t unk_id() const override;
    bool is_fixed_vocab() const override;
    bool is_valid_token_id(int32_t id) const override;

private:
    WordPieceTokenizer(std::unordered_map<std::string, int32_t> token_to_id,
                       std::vector<std::string> id_to_token,
                       int32_t pad_id,
                       int32_t bos_id,
                       int32_t eos_id,
                       int32_t unk_id,
                       bool do_lower_case);

    int32_t lookup_token(const std::string& token) const;
    void encode_wordpiece(const std::string& token, std::vector<int32_t>& ids) const;

    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    int32_t pad_id_ = -1;
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t unk_id_ = -1;
    bool do_lower_case_ = true;
};

/**
 * @brief Lightweight HuggingFace LLaMA tokenizer from tokenizer.json.
 */
class HFLlamaTokenizer final : public Tokenizer {
public:
    static HFLlamaTokenizer from_model_dir(const std::string& hf_model_dir);
    ~HFLlamaTokenizer() override;

    HFLlamaTokenizer(HFLlamaTokenizer&&) noexcept;
    HFLlamaTokenizer& operator=(HFLlamaTokenizer&&) noexcept;
    HFLlamaTokenizer(const HFLlamaTokenizer&) = delete;
    HFLlamaTokenizer& operator=(const HFLlamaTokenizer&) = delete;

    std::vector<int32_t> encode(const std::string& text) override;
    std::string decode(const std::vector<int32_t>& ids) const override;

    int32_t vocab_size() const override;
    int32_t bos_id() const override;
    int32_t eos_id() const override;
    int32_t unk_id() const override;
    bool is_fixed_vocab() const override;
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
