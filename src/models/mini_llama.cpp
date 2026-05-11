#include "tiny_llm/models/mini_llama.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/operators/matmul.h"
#include "tiny_llm/operators/paged_attention.h"
#include "tiny_llm/runtime/execution_context.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tiny_llm {

namespace {

int checked_positive_dim(int64_t dim, const char* name)
{
    if (dim <= 0)
    {
        throw std::runtime_error(std::string("MiniLLaMA::forward_step: ") + name + " must be positive.");
    }
    if (dim > std::numeric_limits<int>::max())
    {
        throw std::runtime_error(std::string("MiniLLaMA::forward_step: ") + name + " is too large.");
    }
    return static_cast<int>(dim);
}

void validate_forward_inputs(const Tensor& input_ids,
                             const Tensor& positions,
                             const Tensor& logits,
                             const MiniLLaMAConfig& cfg)
{
    if (tensor_dtype(input_ids) != DType::kInt32 || tensor_dtype(positions) != DType::kInt32)
    {
        throw std::runtime_error("MiniLLaMA::forward_step: input_ids and positions must be int32.");
    }
    if (tensor_dtype(logits) != DType::kFloat32)
    {
        throw std::runtime_error("MiniLLaMA::forward_step: logits must be float32.");
    }

    const std::vector<int64_t> input_shape = tensor_shape(input_ids);
    const std::vector<int64_t> position_shape = tensor_shape(positions);
    const std::vector<int64_t> logits_shape = tensor_shape(logits);

    if (input_shape.size() != 1 || position_shape.size() != 1)
    {
        throw std::runtime_error("MiniLLaMA::forward_step: input_ids and positions must be rank-1 [B].");
    }
    if (input_shape != position_shape)
    {
        throw std::runtime_error("MiniLLaMA::forward_step: input_ids and positions must have the same shape.");
    }

    if (logits_shape.size() != 2)
    {
        throw std::runtime_error("MiniLLaMA::forward_step: logits must be rank-2 [B, vocab].");
    }

    const int B = checked_positive_dim(input_shape[0], "B");
    const int V = checked_positive_dim(logits_shape[1], "vocab");

    if (logits_shape[0] != input_shape[0])
    {
        throw std::runtime_error("MiniLLaMA::forward_step: logits batch size must match input_ids.");
    }
    if (V != cfg.vocab_size)
    {
        throw std::runtime_error("MiniLLaMA::forward_step: logits vocab size must match config.vocab.");
    }
    if (cfg.hidden_size <= 0 || cfg.vocab_size <= 0)
    {
        throw std::runtime_error("MiniLLaMA::forward_step: model config is invalid.");
    }
    if (tensor_data(input_ids) == nullptr || tensor_data(positions) == nullptr || tensor_data(logits) == nullptr)
    {
        throw std::runtime_error("MiniLLaMA::forward_step: input/output pointers must be non-null.");
    }
    (void)B;
}

void fill_hidden_features(const int32_t* input_ids,
                          const int32_t* positions,
                          int B,
                          int H,
                          std::vector<float>& hidden)
{
    for (int b = 0; b < B; ++b)
    {
        const float token_base = static_cast<float>(input_ids[b] % 997) * 0.01f;
        const float pos_base = static_cast<float>(positions[b] % 509) * 0.001f;
        for (int h = 0; h < H; ++h)
        {
            const float feature = static_cast<float>((h * 31 + 7) % 101) * 0.0005f;
            hidden[static_cast<size_t>(b) * static_cast<size_t>(H) + static_cast<size_t>(h)] =
                token_base + pos_base + feature;
        }
    }
}

void fill_projection_weight(int H, std::vector<float>& w)
{
    for (int i = 0; i < H; ++i)
    {
        for (int j = 0; j < H; ++j)
        {
            float v = 0.0f;
            if (i == j)
            {
                v = 1.0f;
            }
            else if (((i + j) % 29) == 0)
            {
                v = 0.01f;
            }
            w[static_cast<size_t>(i) * static_cast<size_t>(H) + static_cast<size_t>(j)] = v;
        }
    }
}

} // namespace

Tensor MiniLLaMA::forward(const PreparedInputs& inputs, RuntimeContext& ctx)
{
    const Tensor& input_ids = inputs.input_ids;
    const Tensor& positions = inputs.positions;
    Tensor logits = torch::zeros(
        {input_ids.size(0), cfg_.vocab_size},
        torch::TensorOptions().dtype(to_torch_scalar_type(DType::kFloat32)).device(ctx.device()));
    validate_forward_inputs(input_ids, positions, logits, cfg_);

    const std::vector<int64_t> input_shape = tensor_shape(input_ids);
    const int B = checked_positive_dim(input_shape[0], "B");
    const int H = cfg_.hidden_size;
    const int V = cfg_.vocab_size;

    const int32_t* input_ids_ptr = static_cast<const int32_t*>(tensor_data(input_ids));
    const int32_t* positions_ptr = static_cast<const int32_t*>(tensor_data(positions));
    float* logits_ptr = static_cast<float*>(tensor_data(logits));

    std::vector<float> hidden(static_cast<size_t>(B) * static_cast<size_t>(H), 0.0f);
    std::vector<float> proj_w(static_cast<size_t>(H) * static_cast<size_t>(H), 0.0f);
    std::vector<float> hidden_proj(static_cast<size_t>(B) * static_cast<size_t>(H), 0.0f);
    std::vector<float> hidden_attn(static_cast<size_t>(B) * static_cast<size_t>(H), 0.0f);

    fill_hidden_features(input_ids_ptr, positions_ptr, B, H, hidden);
    fill_projection_weight(H, proj_w);

    Tensor hidden_tensor = make_tensor_from_blob(hidden.data(), {B, H}, DType::kFloat32);
    Tensor proj_w_tensor = make_tensor_from_blob(proj_w.data(), {H, H}, DType::kFloat32);
    Tensor hidden_proj_tensor = make_tensor_from_blob(hidden_proj.data(), {B, H}, DType::kFloat32);
    Tensor hidden_attn_tensor = make_tensor_from_blob(hidden_attn.data(), {B, H}, DType::kFloat32);
    ExecutionContext& exec_ctx = resolve_execution_context(ctx.execution());

    ops::gemm(hidden_tensor, proj_w_tensor, hidden_proj_tensor, exec_ctx);
    ops::PagedAttentionRuntimeMetadataGuard metadata_guard(ctx.attention_metadata());
    ops::attention_paged(hidden_proj_tensor, hidden_attn_tensor, exec_ctx);

    for (int b = 0; b < B; ++b)
    {
        float row_mean = 0.0f;
        for (int h = 0; h < H; ++h)
        {
            row_mean += hidden_attn[static_cast<size_t>(b) * static_cast<size_t>(H) + static_cast<size_t>(h)];
        }
        row_mean /= static_cast<float>(H);

        for (int v = 0; v < V; ++v)
        {
            const float feature = hidden_attn[static_cast<size_t>(b) * static_cast<size_t>(H)
                + static_cast<size_t>(v % H)];
            const float periodic = std::sin(static_cast<float>((v + positions_ptr[b]) % 257) * 0.025f);
            logits_ptr[static_cast<size_t>(b) * static_cast<size_t>(V) + static_cast<size_t>(v)] =
                feature + row_mean * 0.1f + periodic * 0.05f;
        }
    }

    return logits;
}

} // namespace tiny_llm
