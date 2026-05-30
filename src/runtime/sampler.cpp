#include "tiny_llm/runtime/sampler.h"

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/sampling.h"

#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace tiny_llm {

namespace {

Tensor tensor_to_cpu_contiguous(const Tensor& tensor)
{
    if (tensor.device().is_cpu())
    {
        return tensor.contiguous();
    }
    return tensor.to(c10::kCPU, /*non_blocking=*/false, /*copy=*/true).contiguous();
}

bool has_active_repetition_penalty(const std::vector<int32_t>& sample_rows,
                                   const std::vector<std::vector<int32_t>>* token_histories,
                                   const std::vector<SamplingParams>* sampling_params)
{
    if (token_histories == nullptr || sampling_params == nullptr)
    {
        return false;
    }
    if (token_histories->size() != sample_rows.size() || sampling_params->size() != sample_rows.size())
    {
        throw std::runtime_error("sample_greedy_rows: repetition penalty metadata size mismatch.");
    }

    for (size_t i = 0; i < sample_rows.size(); ++i)
    {
        if ((*sampling_params)[i].repetition_penalty != 1.0f && !(*token_histories)[i].empty())
        {
            return true;
        }
    }
    return false;
}

std::vector<int64_t> unique_valid_history_tokens(const std::vector<int32_t>& history, int32_t vocab_size)
{
    std::unordered_set<int32_t> seen_tokens;
    seen_tokens.reserve(history.size());
    std::vector<int64_t> unique_tokens;
    unique_tokens.reserve(history.size());
    for (int32_t token_id : history)
    {
        if (token_id < 0 || token_id >= vocab_size)
        {
            throw std::runtime_error("sample_greedy_rows: token history id is out of vocabulary range.");
        }
        if (seen_tokens.insert(token_id).second)
        {
            unique_tokens.push_back(static_cast<int64_t>(token_id));
        }
    }
    return unique_tokens;
}

std::vector<int32_t> sample_cuda_rows_without_repetition_penalty(const Tensor& logits,
                                                                 const std::vector<int32_t>& sample_rows)
{
    bool sample_rows_are_dense_prefix = static_cast<int64_t>(sample_rows.size()) == logits.size(0);
    for (size_t i = 0; i < sample_rows.size() && sample_rows_are_dense_prefix; ++i)
    {
        sample_rows_are_dense_prefix = sample_rows[i] == static_cast<int32_t>(i);
    }

    Tensor sampled_tokens;
    if (sample_rows_are_dense_prefix)
    {
        sampled_tokens = logits.argmax(/*dim=*/1);
    }
    else if (sample_rows.size() == 1)
    {
        sampled_tokens = logits.narrow(0, sample_rows.front(), 1).argmax(/*dim=*/1);
    }
    else
    {
        std::vector<int64_t> row_indices;
        row_indices.reserve(sample_rows.size());
        for (int32_t row : sample_rows)
        {
            row_indices.push_back(static_cast<int64_t>(row));
        }
        Tensor row_tensor = torch::tensor(
            row_indices,
            torch::TensorOptions().dtype(torch::kInt64).device(logits.device()));
        sampled_tokens = logits.index_select(0, row_tensor).argmax(/*dim=*/1);
    }

    Tensor sampled_cpu = sampled_tokens.to(
        torch::TensorOptions().dtype(torch::kInt32).device(c10::kCPU),
        /*non_blocking=*/false,
        /*copy=*/true).contiguous();
    const int32_t* token_ptr = sampled_cpu.data_ptr<int32_t>();
    return std::vector<int32_t>(token_ptr, token_ptr + sample_rows.size());
}

std::vector<int32_t> sample_cuda_rows_with_repetition_penalty(
    const Tensor& logits,
    const std::vector<int32_t>& sample_rows,
    int32_t vocab_size,
    const std::vector<std::vector<int32_t>>& token_histories,
    const std::vector<SamplingParams>& sampling_params)
{
    std::vector<Tensor> sampled_per_row;
    sampled_per_row.reserve(sample_rows.size());
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(logits.device());
    for (size_t sample_index = 0; sample_index < sample_rows.size(); ++sample_index)
    {
        const int32_t row = sample_rows[sample_index];
        const float penalty = sampling_params[sample_index].repetition_penalty;
        if (penalty <= 0.0f)
        {
            throw std::runtime_error("sample_greedy_rows: repetition penalty must be positive.");
        }

        Tensor row_logits = logits.narrow(0, row, 1).squeeze(0);
        if (penalty != 1.0f && !token_histories[sample_index].empty())
        {
            row_logits = row_logits.clone();
            const std::vector<int64_t> history_tokens =
                unique_valid_history_tokens(token_histories[sample_index], vocab_size);
            if (!history_tokens.empty())
            {
                Tensor token_indices = torch::tensor(history_tokens, long_options);
                Tensor history_logits = row_logits.index_select(0, token_indices);
                Tensor adjusted = torch::where(
                    history_logits > 0.0f,
                    history_logits / static_cast<double>(penalty),
                    history_logits * static_cast<double>(penalty));
                row_logits.index_put_({token_indices}, adjusted);
            }
        }
        sampled_per_row.push_back(row_logits.argmax(/*dim=*/0).reshape({1}));
    }

    Tensor sampled_tokens = torch::cat(sampled_per_row, /*dim=*/0);
    Tensor sampled_cpu = sampled_tokens.to(
        torch::TensorOptions().dtype(torch::kInt32).device(c10::kCPU),
        /*non_blocking=*/false,
        /*copy=*/true).contiguous();
    const int32_t* token_ptr = sampled_cpu.data_ptr<int32_t>();
    return std::vector<int32_t>(token_ptr, token_ptr + sample_rows.size());
}

} // namespace

float apply_repetition_penalty_to_logit(float logit, float penalty)
{
    if (penalty <= 0.0f)
    {
        throw std::runtime_error("apply_repetition_penalty_to_logit: penalty must be positive.");
    }
    if (penalty == 1.0f || logit == 0.0f)
    {
        return logit;
    }
    return logit > 0.0f ? logit / penalty : logit * penalty;
}

int32_t sample_argmax_with_repetition_penalty(const float* logits,
                                              int32_t vocab_size,
                                              const std::vector<int32_t>& history,
                                              float penalty)
{
    std::unordered_set<int32_t> seen_tokens;
    seen_tokens.reserve(history.size());
    for (int32_t token_id : history)
    {
        if (token_id < 0 || token_id >= vocab_size)
        {
            throw std::runtime_error("sample_greedy_rows: token history id is out of vocabulary range.");
        }
        seen_tokens.insert(token_id);
    }

    int32_t best_token = 0;
    float best_value = seen_tokens.find(0) == seen_tokens.end()
        ? logits[0]
        : apply_repetition_penalty_to_logit(logits[0], penalty);
    for (int32_t token = 1; token < vocab_size; ++token)
    {
        const float value = seen_tokens.find(token) == seen_tokens.end()
            ? logits[token]
            : apply_repetition_penalty_to_logit(logits[token], penalty);
        if (value > best_value)
        {
            best_value = value;
            best_token = token;
        }
    }
    return best_token;
}

std::vector<int32_t> sample_greedy_rows(const Tensor& logits,
                                        const std::vector<int32_t>& sample_rows,
                                        int32_t vocab_size,
                                        const std::vector<std::vector<int32_t>>* token_histories,
                                        const std::vector<SamplingParams>* sampling_params)
{
    if (!logits.defined())
    {
        throw std::runtime_error("sample_greedy_rows: logits must be defined.");
    }
    if (tensor_dtype(logits) != DType::kFloat32)
    {
        throw std::runtime_error("sample_greedy_rows: logits must be float32.");
    }
    if (logits.dim() != 2 || logits.size(1) != vocab_size)
    {
        throw std::runtime_error("sample_greedy_rows: logits shape must be [rows, vocab_size].");
    }

    std::vector<int32_t> sampled(static_cast<size_t>(logits.size(0)), -1);
    for (int32_t row : sample_rows)
    {
        if (row < 0 || row >= logits.size(0))
        {
            throw std::runtime_error("sample_greedy_rows: sample row is out of range.");
        }
    }

    if (sample_rows.empty())
    {
        return sampled;
    }

    const bool apply_repetition_penalty =
        has_active_repetition_penalty(sample_rows, token_histories, sampling_params);
    if (logits.device().is_cuda())
    {
        const std::vector<int32_t> sampled_tokens = apply_repetition_penalty
            ? sample_cuda_rows_with_repetition_penalty(
                logits,
                sample_rows,
                vocab_size,
                *token_histories,
                *sampling_params)
            : sample_cuda_rows_without_repetition_penalty(logits, sample_rows);
        for (size_t i = 0; i < sample_rows.size(); ++i)
        {
            sampled[static_cast<size_t>(sample_rows[i])] = sampled_tokens[i];
        }
        return sampled;
    }

    Tensor logits_cpu = tensor_to_cpu_contiguous(logits);
    const float* logits_ptr = logits_cpu.data_ptr<float>();
    for (size_t sample_index = 0; sample_index < sample_rows.size(); ++sample_index)
    {
        const int32_t row = sample_rows[sample_index];
        const float* row_logits = logits_ptr + static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
        if (!apply_repetition_penalty || (*sampling_params)[sample_index].repetition_penalty == 1.0f)
        {
            sampled[static_cast<size_t>(row)] = sample_argmax(row_logits, vocab_size);
            continue;
        }

        sampled[static_cast<size_t>(row)] = sample_argmax_with_repetition_penalty(
            row_logits,
            vocab_size,
            (*token_histories)[sample_index],
            (*sampling_params)[sample_index].repetition_penalty);
    }
    return sampled;
}

} // namespace tiny_llm
