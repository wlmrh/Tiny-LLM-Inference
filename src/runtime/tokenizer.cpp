#include "tiny_llm/runtime/tokenizer.h"

#include "../models/hf_json.h"
#include "tiny_llm/models/hf_llama_config_loader.h"
#if defined(__has_include)
#if __has_include(<tokenizers_cpp.h>)
#include <tokenizers_cpp.h>
#else
namespace tokenizers {
class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    virtual std::vector<int32_t> Encode(const std::string& text) = 0;
    virtual std::string Decode(const std::vector<int32_t>& ids) = 0;
    virtual size_t GetVocabSize() = 0;
    virtual std::string IdToToken(int32_t token_id) = 0;
    virtual int32_t TokenToId(const std::string& token) = 0;
    static std::unique_ptr<Tokenizer> FromBlobJSON(const std::string& json_blob);
    static std::unique_ptr<Tokenizer> FromBlobSentencePiece(const std::string& model_blob);
};
} // namespace tokenizers
#endif
#else
namespace tokenizers {
class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    virtual std::vector<int32_t> Encode(const std::string& text) = 0;
    virtual std::string Decode(const std::vector<int32_t>& ids) = 0;
    virtual size_t GetVocabSize() = 0;
    virtual std::string IdToToken(int32_t token_id) = 0;
    virtual int32_t TokenToId(const std::string& token) = 0;
    static std::unique_ptr<Tokenizer> FromBlobJSON(const std::string& json_blob);
    static std::unique_ptr<Tokenizer> FromBlobSentencePiece(const std::string& model_blob);
};
} // namespace tokenizers
#endif

#include <cctype>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace tiny_llm {

namespace {

std::string trim_carriage_return(std::string s)
{
    if (!s.empty() && s.back() == '\r')
    {
        s.pop_back();
    }
    return s;
}

std::string to_lower_ascii(std::string s)
{
    for (char& ch : s)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

bool is_punctuation_char(char ch)
{
    const unsigned char uch = static_cast<unsigned char>(ch);
    return std::ispunct(uch) != 0;
}

bool is_punctuation_token(const std::string& token)
{
    if (token.empty())
    {
        return false;
    }
    if (token.rfind("##", 0) == 0)
    {
        return false;
    }
    return token.size() == 1 && is_punctuation_char(token[0]);
}

std::vector<std::string> basic_pretokenize(const std::string& text, bool do_lower_case)
{
    std::vector<std::string> tokens;
    std::string current;

    for (char raw_ch : text)
    {
        char ch = raw_ch;
        if (do_lower_case)
        {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }

        if (std::isspace(static_cast<unsigned char>(ch)) != 0)
        {
            if (!current.empty())
            {
                tokens.push_back(current);
                current.clear();
            }
            continue;
        }

        if (is_punctuation_char(ch))
        {
            if (!current.empty())
            {
                tokens.push_back(current);
                current.clear();
            }
            tokens.emplace_back(1, ch);
            continue;
        }

        current.push_back(ch);
    }

    if (!current.empty())
    {
        tokens.push_back(current);
    }

    return tokens;
}

std::string read_text_file(const std::string& path, const std::string& error_prefix)
{
    std::ifstream fin(path);
    if (!fin)
    {
        throw std::runtime_error(error_prefix + ": failed to open file: " + path);
    }

    std::ostringstream ss;
    ss << fin.rdbuf();
    return ss.str();
}

std::optional<hf_json::Value> parse_optional_json_file(const std::filesystem::path& path,
                                                       const std::string& error_prefix)
{
    if (!std::filesystem::exists(path))
    {
        return std::nullopt;
    }
    return hf_json::parse(read_text_file(path.string(), error_prefix), error_prefix);
}

std::optional<std::string> read_optional_token_content_field(const hf_json::Value& root,
                                                             const std::string& key,
                                                             const std::string& error_prefix)
{
    const hf_json::Value* value = hf_json::find_object_field(root, key, error_prefix);
    if (value == nullptr)
    {
        return std::nullopt;
    }
    if (value->is_string())
    {
        return value->as_string(error_prefix + ": " + key);
    }

    const hf_json::Value* content = hf_json::find_object_field(*value, "content", error_prefix + ": " + key);
    if (content == nullptr)
    {
        return std::nullopt;
    }
    return content->as_string(error_prefix + ": " + key + ".content");
}

struct HFSpecialTokenStrings {
    std::optional<std::string> bos;
    std::optional<std::string> eos;
    std::optional<std::string> unk;
};

HFSpecialTokenStrings load_hf_special_token_strings(const std::filesystem::path& model_dir,
                                                    const std::string& error_prefix)
{
    HFSpecialTokenStrings out;

    const auto tokenizer_config =
        parse_optional_json_file(model_dir / "tokenizer_config.json", error_prefix + ": tokenizer_config.json");
    if (tokenizer_config.has_value())
    {
        out.bos = read_optional_token_content_field(tokenizer_config.value(), "bos_token", error_prefix);
        out.eos = read_optional_token_content_field(tokenizer_config.value(), "eos_token", error_prefix);
        out.unk = read_optional_token_content_field(tokenizer_config.value(), "unk_token", error_prefix);
    }

    const auto special_tokens_map =
        parse_optional_json_file(model_dir / "special_tokens_map.json", error_prefix + ": special_tokens_map.json");
    if (special_tokens_map.has_value())
    {
        if (!out.bos.has_value())
        {
            out.bos = read_optional_token_content_field(special_tokens_map.value(), "bos_token", error_prefix);
        }
        if (!out.eos.has_value())
        {
            out.eos = read_optional_token_content_field(special_tokens_map.value(), "eos_token", error_prefix);
        }
        if (!out.unk.has_value())
        {
            out.unk = read_optional_token_content_field(special_tokens_map.value(), "unk_token", error_prefix);
        }
    }

    return out;
}

int32_t resolve_special_token_id(const char* token_name,
                                 const std::optional<std::string>& token_string,
                                 int32_t config_id,
                                 tokenizers::Tokenizer* tokenizer,
                                 bool required,
                                 const std::string& error_prefix)
{
    if (tokenizer == nullptr)
    {
        throw std::runtime_error(error_prefix + ": tokenizer must be non-null.");
    }

    if (token_string.has_value())
    {
        const int32_t tokenizer_id = tokenizer->TokenToId(token_string.value());
        if (tokenizer_id >= 0)
        {
            return tokenizer_id;
        }
    }

    if (config_id >= 0)
    {
        return config_id;
    }

    if (required)
    {
        throw std::runtime_error(error_prefix + ": missing required token id for " + std::string(token_name) + ".");
    }

    return -1;
}

} // namespace

TokenizerRegistry::TokenizerRegistry(Tokenizer* tokenizer)
    : tokenizer_(tokenizer)
{
    if (tokenizer_ == nullptr)
    {
        throw std::runtime_error("TokenizerRegistry: tokenizer must be non-null.");
    }
}

WhitespaceTokenizer::WhitespaceTokenizer(int32_t max_vocab_size)
    : max_vocab_size_(max_vocab_size)
{
    if (max_vocab_size_ < 4)
    {
        throw std::runtime_error("WhitespaceTokenizer: max_vocab_size must be >= 4.");
    }

    token_to_id_["<pad>"] = pad_id_;
    token_to_id_["<bos>"] = bos_id_;
    token_to_id_["<eos>"] = eos_id_;
    token_to_id_["<unk>"] = unk_id_;
    id_to_token_.push_back("<pad>");
    id_to_token_.push_back("<bos>");
    id_to_token_.push_back("<eos>");
    id_to_token_.push_back("<unk>");
}

std::vector<int32_t> WhitespaceTokenizer::encode(const std::string& text)
{
    std::vector<int32_t> ids;
    std::istringstream iss(text);
    std::string token;
    while (iss >> token)
    {
        const auto it = token_to_id_.find(token);
        if (it != token_to_id_.end())
        {
            ids.push_back(it->second);
            continue;
        }

        if (static_cast<int32_t>(id_to_token_.size()) >= max_vocab_size_)
        {
            ids.push_back(unk_id_);
            continue;
        }

        const int32_t new_id = static_cast<int32_t>(id_to_token_.size());
        token_to_id_[token] = new_id;
        id_to_token_.push_back(token);
        ids.push_back(new_id);
    }

    if (ids.empty())
    {
        ids.push_back(bos_id_);
    }
    return ids;
}

std::string WhitespaceTokenizer::decode(const std::vector<int32_t>& ids) const
{
    std::string out;
    for (size_t i = 0; i < ids.size(); ++i)
    {
        const int32_t id = ids[i];
        std::string token = "<unk>";
        if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size())
        {
            token = id_to_token_[static_cast<size_t>(id)];
        }

        if (!out.empty())
        {
            out.push_back(' ');
        }
        out += token;
    }
    return out;
}

int32_t WhitespaceTokenizer::vocab_size() const
{
    return max_vocab_size_;
}

int32_t WhitespaceTokenizer::bos_id() const
{
    return bos_id_;
}

int32_t WhitespaceTokenizer::eos_id() const
{
    return eos_id_;
}

int32_t WhitespaceTokenizer::unk_id() const
{
    return unk_id_;
}

bool WhitespaceTokenizer::is_fixed_vocab() const
{
    return false;
}

bool WhitespaceTokenizer::is_valid_token_id(int32_t id) const
{
    return id >= 0 && id < max_vocab_size_;
}

WordPieceTokenizer WordPieceTokenizer::from_vocab_file(const std::string& path,
                                                       const std::string& pad_token,
                                                       const std::string& bos_token,
                                                       const std::string& eos_token,
                                                       const std::string& unk_token,
                                                       bool do_lower_case)
{
    std::ifstream fin(path);
    if (!fin)
    {
        throw std::runtime_error("WordPieceTokenizer::from_vocab_file: failed to open vocab file: " + path);
    }

    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<std::string> id_to_token;

    std::string line;
    while (std::getline(fin, line))
    {
        line = trim_carriage_return(line);
        if (line.empty())
        {
            continue;
        }

        if (token_to_id.find(line) != token_to_id.end())
        {
            throw std::runtime_error("WordPieceTokenizer::from_vocab_file: duplicated token in vocab: " + line);
        }

        const int32_t id = static_cast<int32_t>(id_to_token.size());
        token_to_id[line] = id;
        id_to_token.push_back(line);
    }

    if (id_to_token.empty())
    {
        throw std::runtime_error("WordPieceTokenizer::from_vocab_file: vocab is empty.");
    }

    const auto find_id = [&](const std::string& token_name) -> int32_t {
        const auto it = token_to_id.find(token_name);
        if (it == token_to_id.end())
        {
            throw std::runtime_error("WordPieceTokenizer::from_vocab_file: missing special token: " + token_name);
        }
        return it->second;
    };

    const int32_t pad_id = find_id(pad_token);
    const int32_t bos_id = find_id(bos_token);
    const int32_t eos_id = find_id(eos_token);
    const int32_t unk_id = find_id(unk_token);

    return WordPieceTokenizer(std::move(token_to_id),
                              std::move(id_to_token),
                              pad_id,
                              bos_id,
                              eos_id,
                              unk_id,
                              do_lower_case);
}

std::vector<int32_t> WordPieceTokenizer::encode(const std::string& text)
{
    std::vector<int32_t> ids;
    const std::vector<std::string> tokens = basic_pretokenize(text, do_lower_case_);
    for (const std::string& token : tokens)
    {
        encode_wordpiece(token, ids);
    }

    if (ids.empty())
    {
        ids.push_back(bos_id_);
    }
    return ids;
}

std::string WordPieceTokenizer::decode(const std::vector<int32_t>& ids) const
{
    std::string out;
    for (int32_t id : ids)
    {
        if (!is_valid_token_id(id))
        {
            id = unk_id_;
        }

        if (id == pad_id_ || id == bos_id_ || id == eos_id_)
        {
            continue;
        }

        const std::string& token = id_to_token_[static_cast<size_t>(id)];
        if (token.rfind("##", 0) == 0)
        {
            out += token.substr(2);
            continue;
        }

        if (out.empty())
        {
            out += token;
            continue;
        }

        if (is_punctuation_token(token))
        {
            out += token;
        }
        else
        {
            out.push_back(' ');
            out += token;
        }
    }
    return out;
}

int32_t WordPieceTokenizer::vocab_size() const
{
    return static_cast<int32_t>(id_to_token_.size());
}

int32_t WordPieceTokenizer::bos_id() const
{
    return bos_id_;
}

int32_t WordPieceTokenizer::eos_id() const
{
    return eos_id_;
}

int32_t WordPieceTokenizer::unk_id() const
{
    return unk_id_;
}

bool WordPieceTokenizer::is_fixed_vocab() const
{
    return true;
}

bool WordPieceTokenizer::is_valid_token_id(int32_t id) const
{
    return id >= 0 && static_cast<size_t>(id) < id_to_token_.size();
}

WordPieceTokenizer::WordPieceTokenizer(std::unordered_map<std::string, int32_t> token_to_id,
                                       std::vector<std::string> id_to_token,
                                       int32_t pad_id,
                                       int32_t bos_id,
                                       int32_t eos_id,
                                       int32_t unk_id,
                                       bool do_lower_case)
    : token_to_id_(std::move(token_to_id)),
      id_to_token_(std::move(id_to_token)),
      pad_id_(pad_id),
      bos_id_(bos_id),
      eos_id_(eos_id),
      unk_id_(unk_id),
      do_lower_case_(do_lower_case)
{
    if (!is_valid_token_id(pad_id_) || !is_valid_token_id(bos_id_) || !is_valid_token_id(eos_id_) || !is_valid_token_id(unk_id_))
    {
        throw std::runtime_error("WordPieceTokenizer: special token id is out of vocab range.");
    }
}

int32_t WordPieceTokenizer::lookup_token(const std::string& token) const
{
    const auto it = token_to_id_.find(token);
    if (it == token_to_id_.end())
    {
        return -1;
    }
    return it->second;
}

void WordPieceTokenizer::encode_wordpiece(const std::string& token, std::vector<int32_t>& ids) const
{
    std::string normalized = token;
    if (do_lower_case_)
    {
        normalized = to_lower_ascii(normalized);
    }

    const int32_t direct = lookup_token(normalized);
    if (direct >= 0)
    {
        ids.push_back(direct);
        return;
    }

    size_t start = 0;
    std::vector<int32_t> pieces;
    while (start < normalized.size())
    {
        bool matched = false;
        size_t end = normalized.size();
        while (end > start)
        {
            std::string piece = normalized.substr(start, end - start);
            if (start > 0)
            {
                piece = "##" + piece;
            }

            const int32_t id = lookup_token(piece);
            if (id >= 0)
            {
                pieces.push_back(id);
                start = end;
                matched = true;
                break;
            }
            --end;
        }

        if (!matched)
        {
            ids.push_back(unk_id_);
            return;
        }
    }

    ids.insert(ids.end(), pieces.begin(), pieces.end());
}

struct HFLlamaTokenizer::Impl {
    Impl(std::unique_ptr<tokenizers::Tokenizer> tokenizer_handle,
         int32_t vocab,
         int32_t bos,
         int32_t eos,
         int32_t unk)
        : tokenizer(std::move(tokenizer_handle)),
          vocab_size(vocab),
          bos_id(bos),
          eos_id(eos),
          unk_id(unk)
    {
        if (!tokenizer)
        {
            throw std::runtime_error("HFLlamaTokenizer: tokenizer handle must be non-null.");
        }
        if (vocab_size <= 0)
        {
            throw std::runtime_error("HFLlamaTokenizer: vocab size must be positive.");
        }
        if (bos_id < 0 || bos_id >= vocab_size
            || eos_id < 0 || eos_id >= vocab_size
            || (unk_id >= 0 && unk_id >= vocab_size))
        {
            throw std::runtime_error("HFLlamaTokenizer: special token id is out of vocab range.");
        }
    }

    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    int32_t vocab_size = -1;
    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t unk_id = -1;
};

HFLlamaTokenizer HFLlamaTokenizer::from_model_dir(const std::string& hf_model_dir)
{
    constexpr const char* kErr = "HFLlamaTokenizer::from_model_dir";
    if (hf_model_dir.empty())
    {
        throw std::runtime_error(std::string(kErr) + ": hf_model_dir must be non-empty.");
    }
    const std::filesystem::path model_dir(hf_model_dir);
    const std::filesystem::path tokenizer_json_path = model_dir / "tokenizer.json";
    const std::filesystem::path tokenizer_spm_path = model_dir / "tokenizer.model";

    std::unique_ptr<tokenizers::Tokenizer> tokenizer_handle;
    if (std::filesystem::exists(tokenizer_json_path))
    {
        tokenizer_handle = tokenizers::Tokenizer::FromBlobJSON(
            read_text_file(tokenizer_json_path.string(), kErr));
    }
    else if (std::filesystem::exists(tokenizer_spm_path))
    {
        tokenizer_handle = tokenizers::Tokenizer::FromBlobSentencePiece(
            read_text_file(tokenizer_spm_path.string(), kErr));
    }
    else
    {
        throw std::runtime_error(std::string(kErr)
                                 + ": neither tokenizer.json nor tokenizer.model exists in "
                                 + hf_model_dir);
    }

    if (tokenizer_handle == nullptr)
    {
        throw std::runtime_error(std::string(kErr) + ": failed to construct tokenizer via tokenizers-cpp.");
    }

    const LlamaConfig cfg = HFLlamaConfigLoader::load_from_dir(hf_model_dir);
    const HFSpecialTokenStrings token_strings = load_hf_special_token_strings(model_dir, kErr);
    const int32_t vocab = static_cast<int32_t>(tokenizer_handle->GetVocabSize());
    const int32_t bos_id =
        resolve_special_token_id("bos_token_id", token_strings.bos, cfg.bos_token_id, tokenizer_handle.get(), true, kErr);
    const int32_t eos_id =
        resolve_special_token_id("eos_token_id", token_strings.eos, cfg.eos_token_id, tokenizer_handle.get(), true, kErr);
    const int32_t unk_id =
        resolve_special_token_id("unk_token_id", token_strings.unk, cfg.unk_token_id, tokenizer_handle.get(), false, kErr);

    auto impl = std::make_unique<Impl>(std::move(tokenizer_handle), vocab, bos_id, eos_id, unk_id);
    return HFLlamaTokenizer(std::move(impl));
}

std::vector<int32_t> HFLlamaTokenizer::encode(const std::string& text)
{
    std::vector<int32_t> ids = impl_->tokenizer->Encode(text);
    if (ids.empty() && bos_id_ >= 0)
    {
        ids.push_back(bos_id_);
    }
    return ids;
}

std::string HFLlamaTokenizer::decode(const std::vector<int32_t>& ids) const
{
    return impl_->tokenizer->Decode(ids);
}

int32_t HFLlamaTokenizer::vocab_size() const
{
    return impl_->vocab_size;
}

int32_t HFLlamaTokenizer::bos_id() const
{
    return bos_id_;
}

int32_t HFLlamaTokenizer::eos_id() const
{
    return eos_id_;
}

int32_t HFLlamaTokenizer::unk_id() const
{
    return unk_id_;
}

bool HFLlamaTokenizer::is_fixed_vocab() const
{
    return true;
}

bool HFLlamaTokenizer::is_valid_token_id(int32_t id) const
{
    return id >= 0 && id < vocab_size();
}

HFLlamaTokenizer::HFLlamaTokenizer(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
    if (!impl_)
    {
        throw std::runtime_error("HFLlamaTokenizer: impl must be non-null.");
    }
    bos_id_ = impl_->bos_id;
    eos_id_ = impl_->eos_id;
    unk_id_ = impl_->unk_id;
}

HFLlamaTokenizer::~HFLlamaTokenizer() = default;
HFLlamaTokenizer::HFLlamaTokenizer(HFLlamaTokenizer&&) noexcept = default;
HFLlamaTokenizer& HFLlamaTokenizer::operator=(HFLlamaTokenizer&&) noexcept = default;

} // namespace tiny_llm
