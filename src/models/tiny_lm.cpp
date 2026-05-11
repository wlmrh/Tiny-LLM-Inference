#include "tiny_llm/models/tiny_lm.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/operators/matmul.h"
#include "tiny_llm/runtime/execution_context.h"

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tiny_llm {

namespace {

std::string read_token(std::istream& in, const char* error_prefix)
{
    std::string token;
    while (in >> token)
    {
        if (!token.empty() && token[0] == '#')
        {
            std::string ignored;
            std::getline(in, ignored);
            continue;
        }
        return token;
    }
    throw std::runtime_error(std::string(error_prefix) + ": unexpected end of checkpoint.");
}

int32_t read_int32(std::istream& in, const char* error_prefix)
{
    const std::string token = read_token(in, error_prefix);
    long long value = 0;
    try
    {
        value = std::stoll(token);
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(std::string(error_prefix) + ": invalid integer token: " + token);
    }

    if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
    {
        throw std::runtime_error(std::string(error_prefix) + ": integer out of int32 range.");
    }
    return static_cast<int32_t>(value);
}

float read_float(std::istream& in, const char* error_prefix)
{
    const std::string token = read_token(in, error_prefix);
    try
    {
        return std::stof(token);
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(std::string(error_prefix) + ": invalid float token: " + token);
    }
}

void expect_token(std::istream& in, const std::string& expected, const char* error_prefix)
{
    const std::string token = read_token(in, error_prefix);
    if (token != expected)
    {
        throw std::runtime_error(std::string(error_prefix) + ": expected token '" + expected + "' but got '" + token + "'.");
    }
}

std::vector<float> read_float_array(std::istream& in, int32_t count, const char* error_prefix)
{
    if (count <= 0)
    {
        throw std::runtime_error(std::string(error_prefix) + ": tensor element count must be positive.");
    }

    std::vector<float> values(static_cast<size_t>(count), 0.0f);
    for (int32_t i = 0; i < count; ++i)
    {
        values[static_cast<size_t>(i)] = read_float(in, error_prefix);
    }
    return values;
}

int32_t checked_positive_dim(int64_t dim, const char* name)
{
    if (dim <= 0)
    {
        throw std::runtime_error(std::string("TinyEmbeddingLM::forward_step: ") + name + " must be positive.");
    }
    if (dim > std::numeric_limits<int32_t>::max())
    {
        throw std::runtime_error(std::string("TinyEmbeddingLM::forward_step: ") + name + " is too large.");
    }
    return static_cast<int32_t>(dim);
}

void validate_model_buffers(const TinyLMConfig& cfg,
                            const std::vector<float>& embedding,
                            const std::vector<float>& projection,
                            const std::vector<float>& bias)
{
    if (cfg.vocab <= 0 || cfg.hidden <= 0 || cfg.num_layers <= 0)
    {
        throw std::runtime_error("TinyEmbeddingLM: config dimensions must be positive.");
    }

    const int64_t embedding_expected = static_cast<int64_t>(cfg.vocab) * static_cast<int64_t>(cfg.hidden);
    const int64_t projection_expected = static_cast<int64_t>(cfg.hidden) * static_cast<int64_t>(cfg.vocab);
    const int64_t bias_expected = static_cast<int64_t>(cfg.vocab);

    if (static_cast<int64_t>(embedding.size()) != embedding_expected)
    {
        throw std::runtime_error("TinyEmbeddingLM: embedding size mismatches config.");
    }
    if (static_cast<int64_t>(projection.size()) != projection_expected)
    {
        throw std::runtime_error("TinyEmbeddingLM: projection size mismatches config.");
    }
    if (static_cast<int64_t>(bias.size()) != bias_expected)
    {
        throw std::runtime_error("TinyEmbeddingLM: bias size mismatches config.");
    }

    if (cfg.bos_id < 0 || cfg.bos_id >= cfg.vocab ||
        cfg.eos_id < 0 || cfg.eos_id >= cfg.vocab ||
        cfg.unk_id < 0 || cfg.unk_id >= cfg.vocab)
    {
        throw std::runtime_error("TinyEmbeddingLM: special token ids are out of vocab range.");
    }
}

void validate_forward_inputs(const Tensor& input_ids,
                             const Tensor& positions,
                             const Tensor& logits,
                             const TinyLMConfig& cfg)
{
    if (tensor_dtype(input_ids) != DType::kInt32 || tensor_dtype(positions) != DType::kInt32)
    {
        throw std::runtime_error("TinyEmbeddingLM::forward_step: input_ids and positions must be int32.");
    }
    if (tensor_dtype(logits) != DType::kFloat32)
    {
        throw std::runtime_error("TinyEmbeddingLM::forward_step: logits must be float32.");
    }

    const std::vector<int64_t> input_shape = tensor_shape(input_ids);
    const std::vector<int64_t> position_shape = tensor_shape(positions);
    const std::vector<int64_t> logits_shape = tensor_shape(logits);

    if (input_shape.size() != 1 || position_shape.size() != 1)
    {
        throw std::runtime_error("TinyEmbeddingLM::forward_step: input_ids and positions must be rank-1 [B].");
    }
    if (input_shape != position_shape)
    {
        throw std::runtime_error("TinyEmbeddingLM::forward_step: input_ids and positions must have the same shape.");
    }

    if (logits_shape.size() != 2)
    {
        throw std::runtime_error("TinyEmbeddingLM::forward_step: logits must be rank-2 [B, vocab].");
    }

    const int32_t B = checked_positive_dim(input_shape[0], "B");
    const int32_t V = checked_positive_dim(logits_shape[1], "vocab");

    if (logits_shape[0] != input_shape[0])
    {
        throw std::runtime_error("TinyEmbeddingLM::forward_step: logits batch size must match input_ids.");
    }
    if (V != cfg.vocab)
    {
        throw std::runtime_error("TinyEmbeddingLM::forward_step: logits vocab size must match config.vocab.");
    }
    if (tensor_data(input_ids) == nullptr || tensor_data(positions) == nullptr || tensor_data(logits) == nullptr)
    {
        throw std::runtime_error("TinyEmbeddingLM::forward_step: input/output pointers must be non-null.");
    }

    (void)B;
}

} // namespace

TinyEmbeddingLM::TinyEmbeddingLM(TinyLMConfig cfg,
                                 std::vector<float> embedding,
                                 std::vector<float> projection,
                                 std::vector<float> bias)
    : cfg_(cfg),
      embedding_(std::move(embedding)),
      projection_(std::move(projection)),
      bias_(std::move(bias))
{
    validate_model_buffers(cfg_, embedding_, projection_, bias_);
}

TinyEmbeddingLM TinyEmbeddingLM::from_checkpoint(const std::string& path)
{
    constexpr const char* kErr = "TinyEmbeddingLM::from_checkpoint";

    std::ifstream fin(path);
    if (!fin)
    {
        throw std::runtime_error(std::string(kErr) + ": failed to open checkpoint: " + path);
    }

    expect_token(fin, "tiny_lm_checkpoint_v1", kErr);

    TinyLMConfig cfg;
    expect_token(fin, "vocab_size", kErr);
    cfg.vocab = read_int32(fin, kErr);

    expect_token(fin, "hidden_size", kErr);
    cfg.hidden = read_int32(fin, kErr);

    expect_token(fin, "num_layers", kErr);
    cfg.num_layers = read_int32(fin, kErr);

    expect_token(fin, "bos_id", kErr);
    cfg.bos_id = read_int32(fin, kErr);

    expect_token(fin, "eos_id", kErr);
    cfg.eos_id = read_int32(fin, kErr);

    expect_token(fin, "unk_id", kErr);
    cfg.unk_id = read_int32(fin, kErr);

    const int64_t embedding_expected = static_cast<int64_t>(cfg.vocab) * static_cast<int64_t>(cfg.hidden);
    if (embedding_expected <= 0 || embedding_expected > std::numeric_limits<int32_t>::max())
    {
        throw std::runtime_error(std::string(kErr) + ": invalid embedding tensor size from config.");
    }
    expect_token(fin, "embedding", kErr);
    const int32_t embedding_count = read_int32(fin, kErr);
    if (embedding_count != static_cast<int32_t>(embedding_expected))
    {
        throw std::runtime_error(std::string(kErr) + ": embedding element count mismatches config.");
    }
    std::vector<float> embedding = read_float_array(fin, embedding_count, kErr);

    const int64_t projection_expected = static_cast<int64_t>(cfg.hidden) * static_cast<int64_t>(cfg.vocab);
    if (projection_expected <= 0 || projection_expected > std::numeric_limits<int32_t>::max())
    {
        throw std::runtime_error(std::string(kErr) + ": invalid projection tensor size from config.");
    }
    expect_token(fin, "projection", kErr);
    const int32_t projection_count = read_int32(fin, kErr);
    if (projection_count != static_cast<int32_t>(projection_expected))
    {
        throw std::runtime_error(std::string(kErr) + ": projection element count mismatches config.");
    }
    std::vector<float> projection = read_float_array(fin, projection_count, kErr);

    expect_token(fin, "bias", kErr);
    const int32_t bias_count = read_int32(fin, kErr);
    if (bias_count != cfg.vocab)
    {
        throw std::runtime_error(std::string(kErr) + ": bias element count mismatches config.vocab.");
    }
    std::vector<float> bias = read_float_array(fin, bias_count, kErr);

    return TinyEmbeddingLM(cfg, std::move(embedding), std::move(projection), std::move(bias));
}

Tensor TinyEmbeddingLM::forward(const PreparedInputs& inputs, RuntimeContext& ctx)
{
    const Tensor& input_ids = inputs.input_ids;
    const Tensor& positions = inputs.positions;
    Tensor logits = torch::zeros(
        {input_ids.size(0), cfg_.vocab},
        torch::TensorOptions().dtype(to_torch_scalar_type(DType::kFloat32)).device(ctx.device()));
    validate_forward_inputs(input_ids, positions, logits, cfg_);

    const std::vector<int64_t> input_shape = tensor_shape(input_ids);
    const int32_t B = checked_positive_dim(input_shape[0], "B");
    const int32_t H = cfg_.hidden;
    const int32_t V = cfg_.vocab;

    const int32_t* input_ptr = static_cast<const int32_t*>(tensor_data(input_ids));
    float* logits_ptr = static_cast<float*>(tensor_data(logits));

    std::vector<float> hidden(static_cast<size_t>(B) * static_cast<size_t>(H), 0.0f);
    for (int32_t b = 0; b < B; ++b)
    {
        int32_t token_id = input_ptr[b];
        if (token_id < 0 || token_id >= V)
        {
            token_id = cfg_.unk_id;
        }

        const size_t emb_row_offset = static_cast<size_t>(token_id) * static_cast<size_t>(H);
        const size_t hidden_row_offset = static_cast<size_t>(b) * static_cast<size_t>(H);
        for (int32_t h = 0; h < H; ++h)
        {
            hidden[hidden_row_offset + static_cast<size_t>(h)] = embedding_[emb_row_offset + static_cast<size_t>(h)];
        }
    }

    Tensor hidden_tensor = make_tensor_from_blob(hidden.data(), {B, H}, DType::kFloat32);
    Tensor projection_tensor = make_tensor_from_blob(const_cast<float*>(projection_.data()), {H, V}, DType::kFloat32);
    ExecutionContext& exec_ctx = resolve_execution_context(ctx.execution());
    ops::gemm(hidden_tensor, projection_tensor, logits, exec_ctx);

    for (int32_t b = 0; b < B; ++b)
    {
        const size_t row_offset = static_cast<size_t>(b) * static_cast<size_t>(V);
        for (int32_t v = 0; v < V; ++v)
        {
            logits_ptr[row_offset + static_cast<size_t>(v)] += bias_[static_cast<size_t>(v)];
        }
    }

    return logits;
}

} // namespace tiny_llm
