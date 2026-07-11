#include "tiny_llm/runtime/sampler.h"

#include "tiny_llm/core/tensor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#if TINYLLM_ENABLE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#endif

namespace tiny_llm
{

#if TINYLLM_ENABLE_CUDA
namespace cuda
{
void launch_mark_repetition_history_mask(const int64_t *history_tokens, const int32_t *history_offsets,
                                         uint8_t *history_mask, int32_t sample_count, int32_t vocab_size,
                                         cudaStream_t stream);
void launch_argmax_repetition_penalty_f32(const float *logits, const int32_t *sample_rows, const uint8_t *history_mask,
                                          const float *penalties, int32_t *sampled_tokens, int32_t sample_count,
                                          int32_t vocab_size, cudaStream_t stream);
} // namespace cuda
#endif

namespace
{

#if TINYLLM_ENABLE_CUDA
struct CudaSamplerScratch
{
    bool initialized = false;
    c10::Device device = c10::Device(c10::kCPU);
    int32_t sample_capacity = 0;
    int32_t history_capacity = 0;
    int32_t vocab_size = 0;
    Tensor sample_rows;
    Tensor history_offsets;
    Tensor history_tokens;
    Tensor penalties;
    Tensor history_mask;
    Tensor sampled_tokens;
};

thread_local CudaSamplerScratch g_cuda_sampler_scratch;

int32_t grow_capacity(int32_t current, int32_t required)
{
    int32_t capacity = std::max(current, 1);
    while (capacity < required)
    {
        capacity *= 2;
    }
    return capacity;
}

CudaSamplerScratch &get_cuda_sampler_scratch(c10::Device device, int32_t sample_count, int32_t vocab_size,
                                             int32_t history_count)
{
    CudaSamplerScratch &scratch = g_cuda_sampler_scratch;
    const bool needs_reallocate = !scratch.initialized || scratch.device != device ||
                                  scratch.sample_capacity < sample_count || scratch.vocab_size != vocab_size;
    if (needs_reallocate)
    {
        scratch.initialized = true;
        scratch.device = device;
        scratch.sample_capacity = grow_capacity(scratch.sample_capacity, sample_count);
        scratch.vocab_size = vocab_size;
        const auto int32_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
        const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        scratch.sample_rows = torch::empty({scratch.sample_capacity}, int32_options);
        scratch.history_offsets = torch::empty({scratch.sample_capacity + 1}, int32_options);
        scratch.penalties = torch::empty({scratch.sample_capacity}, float_options);
        scratch.history_mask = torch::empty({scratch.sample_capacity, vocab_size},
                                            torch::TensorOptions().dtype(torch::kUInt8).device(device));
        scratch.sampled_tokens = torch::empty({scratch.sample_capacity}, int32_options);
    }

    if (scratch.history_capacity < history_count)
    {
        scratch.history_capacity = grow_capacity(scratch.history_capacity, history_count);
        scratch.history_tokens =
            torch::empty({scratch.history_capacity}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    }
    return scratch;
}

template <typename T> Tensor vector_cpu_view(const std::vector<T> &values, c10::ScalarType dtype)
{
    return torch::from_blob(const_cast<T *>(values.data()), {static_cast<int64_t>(values.size())},
                            torch::TensorOptions().dtype(dtype).device(c10::kCPU));
}
#endif

Tensor tensor_to_cpu_contiguous(const Tensor &tensor)
{
    if (tensor.device().is_cpu())
    {
        return tensor.contiguous();
    }
    return tensor.to(c10::kCPU, /*non_blocking=*/false, /*copy=*/true).contiguous();
}

void validate_sampling_params(const SamplingParams &params)
{
    if (params.temperature < 0.0f)
    {
        throw std::runtime_error("sample_rows: temperature must be >= 0.");
    }
    if (!(params.top_p > 0.0f && params.top_p <= 1.0f))
    {
        throw std::runtime_error("sample_rows: top_p must be in (0, 1].");
    }
    if (params.top_k < 0)
    {
        throw std::runtime_error("sample_rows: top_k must be >= 0.");
    }
    if (params.repetition_penalty <= 0.0f)
    {
        throw std::runtime_error("sample_rows: repetition_penalty must be positive.");
    }
}

void validate_sampling_metadata(const std::vector<int32_t> &sample_rows,
                                const std::vector<std::vector<int32_t>> *token_histories,
                                const std::vector<SamplingParams> *sampling_params,
                                const std::vector<uint64_t> *request_ids)
{
    if (token_histories != nullptr && token_histories->size() != sample_rows.size())
    {
        throw std::runtime_error("sample_rows: token history metadata size mismatch.");
    }
    if (sampling_params != nullptr)
    {
        if (sampling_params->size() != sample_rows.size())
        {
            throw std::runtime_error("sample_rows: sampling parameter metadata size mismatch.");
        }
        for (const SamplingParams &params : *sampling_params)
        {
            validate_sampling_params(params);
        }
    }
    if (request_ids != nullptr && request_ids->size() != sample_rows.size())
    {
        throw std::runtime_error("sample_rows: request id metadata size mismatch.");
    }
}

bool has_active_repetition_penalty(const std::vector<int32_t> &sample_rows,
                                   const std::vector<std::vector<int32_t>> *token_histories,
                                   const std::vector<SamplingParams> *sampling_params)
{
    if (token_histories == nullptr || sampling_params == nullptr)
    {
        return false;
    }
    if (token_histories->size() != sample_rows.size() || sampling_params->size() != sample_rows.size())
    {
        throw std::runtime_error("sample_rows: repetition penalty metadata size mismatch.");
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

bool has_non_greedy_sampling(const std::vector<int32_t> &sample_rows,
                             const std::vector<SamplingParams> *sampling_params)
{
    if (sampling_params == nullptr)
    {
        return false;
    }
    if (sampling_params->size() != sample_rows.size())
    {
        throw std::runtime_error("sample_rows: sampling parameter metadata size mismatch.");
    }
    for (const SamplingParams &params : *sampling_params)
    {
        if (params.temperature > 0.0f || params.top_k > 0 || params.top_p < 1.0f)
        {
            return true;
        }
    }
    return false;
}

std::vector<int64_t> unique_valid_history_tokens(const std::vector<int32_t> &history, int32_t vocab_size)
{
    std::unordered_set<int32_t> seen_tokens;
    seen_tokens.reserve(history.size());
    std::vector<int64_t> unique_tokens;
    unique_tokens.reserve(history.size());
    for (int32_t token_id : history)
    {
        if (token_id < 0 || token_id >= vocab_size)
        {
            throw std::runtime_error("sample_rows: token history id is out of vocabulary range.");
        }
        if (seen_tokens.insert(token_id).second)
        {
            unique_tokens.push_back(static_cast<int64_t>(token_id));
        }
    }
    return unique_tokens;
}

uint64_t mix_seed(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

uint64_t derive_sample_seed(const SamplingParams &params, uint64_t request_id, size_t history_size, size_t sample_index)
{
    uint64_t seed = mix_seed(params.seed);
    seed ^= mix_seed(request_id + 0x100000001b3ULL);
    seed ^= mix_seed(static_cast<uint64_t>(history_size) + 0x9e3779b97f4a7c15ULL);
    seed ^= mix_seed(static_cast<uint64_t>(sample_index) + 0xbf58476d1ce4e5b9ULL);
    return seed;
}

int32_t argmax_row(const std::vector<float> &row)
{
    int32_t best = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < static_cast<int32_t>(row.size()); ++token)
    {
        if (row[static_cast<size_t>(token)] > best_value)
        {
            best_value = row[static_cast<size_t>(token)];
            best = token;
        }
    }
    if (best < 0 || best_value == -std::numeric_limits<float>::infinity())
    {
        throw std::runtime_error("sample_rows: no valid token remains after filtering.");
    }
    return best;
}

float apply_repetition_penalty_to_logit(float logit, float penalty)
{
    if (penalty <= 0.0f)
    {
        throw std::runtime_error("sample_rows: repetition_penalty must be positive.");
    }
    if (penalty == 1.0f || logit == 0.0f)
    {
        return logit;
    }
    return logit > 0.0f ? logit / penalty : logit * penalty;
}

void apply_repetition_penalty(std::vector<float> &row, const std::vector<int32_t> &history, int32_t vocab_size,
                              float penalty)
{
    const std::vector<int64_t> history_tokens = unique_valid_history_tokens(history, vocab_size);
    for (int64_t token : history_tokens)
    {
        float &value = row[static_cast<size_t>(token)];
        value = apply_repetition_penalty_to_logit(value, penalty);
    }
}

void apply_top_k_filter(std::vector<float> &row, int32_t top_k)
{
    if (top_k <= 0 || top_k >= static_cast<int32_t>(row.size()))
    {
        return;
    }

    std::vector<int32_t> indices(row.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                      [&](int32_t lhs, int32_t rhs)
                      {
                          const float l = row[static_cast<size_t>(lhs)];
                          const float r = row[static_cast<size_t>(rhs)];
                          if (l == r)
                          {
                              return lhs < rhs;
                          }
                          return l > r;
                      });

    std::vector<bool> keep(row.size(), false);
    for (int32_t i = 0; i < top_k; ++i)
    {
        keep[static_cast<size_t>(indices[static_cast<size_t>(i)])] = true;
    }
    for (size_t token = 0; token < row.size(); ++token)
    {
        if (!keep[token])
        {
            row[token] = -std::numeric_limits<float>::infinity();
        }
    }
}

std::vector<double> softmax_weights(const std::vector<float> &row, float temperature)
{
    const double temp = temperature > 0.0f ? static_cast<double>(temperature) : 1.0;
    double max_value = -std::numeric_limits<double>::infinity();
    for (float value : row)
    {
        if (value != -std::numeric_limits<float>::infinity())
        {
            max_value = std::max(max_value, static_cast<double>(value) / temp);
        }
    }
    if (max_value == -std::numeric_limits<double>::infinity())
    {
        throw std::runtime_error("sample_rows: no valid logits remain after filtering.");
    }

    std::vector<double> weights(row.size(), 0.0);
    for (size_t token = 0; token < row.size(); ++token)
    {
        if (row[token] != -std::numeric_limits<float>::infinity())
        {
            weights[token] = std::exp(static_cast<double>(row[token]) / temp - max_value);
        }
    }
    return weights;
}

void apply_top_p_filter(std::vector<float> &row, float top_p, float temperature)
{
    if (top_p >= 1.0f)
    {
        return;
    }

    std::vector<double> weights = softmax_weights(row, temperature);
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total <= 0.0)
    {
        throw std::runtime_error("sample_rows: probability mass is zero.");
    }
    for (double &weight : weights)
    {
        weight /= total;
    }

    std::vector<int32_t> indices(row.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int32_t lhs, int32_t rhs)
              {
                  const double l = weights[static_cast<size_t>(lhs)];
                  const double r = weights[static_cast<size_t>(rhs)];
                  if (l == r)
                  {
                      return lhs < rhs;
                  }
                  return l > r;
              });

    std::vector<bool> keep(row.size(), false);
    double cumulative = 0.0;
    bool kept_any = false;
    for (int32_t token : indices)
    {
        if (weights[static_cast<size_t>(token)] <= 0.0)
        {
            continue;
        }
        keep[static_cast<size_t>(token)] = true;
        kept_any = true;
        cumulative += weights[static_cast<size_t>(token)];
        if (cumulative >= static_cast<double>(top_p))
        {
            break;
        }
    }
    if (!kept_any)
    {
        keep[static_cast<size_t>(indices.front())] = true;
    }

    for (size_t token = 0; token < row.size(); ++token)
    {
        if (!keep[token])
        {
            row[token] = -std::numeric_limits<float>::infinity();
        }
    }
}

int32_t sample_filtered_row(std::vector<float> row, const std::vector<int32_t> &history, int32_t vocab_size,
                            const SamplingParams &params, uint64_t request_id, size_t sample_index)
{
    validate_sampling_params(params);
    if (params.repetition_penalty != 1.0f && !history.empty())
    {
        apply_repetition_penalty(row, history, vocab_size, params.repetition_penalty);
    }
    apply_top_k_filter(row, params.top_k);
    apply_top_p_filter(row, params.top_p, params.temperature);

    if (params.temperature == 0.0f)
    {
        return argmax_row(row);
    }

    std::vector<double> weights = softmax_weights(row, params.temperature);
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total <= 0.0)
    {
        throw std::runtime_error("sample_rows: probability mass is zero.");
    }

    std::mt19937_64 rng(derive_sample_seed(params, request_id, history.size(), sample_index));
    std::discrete_distribution<int32_t> dist(weights.begin(), weights.end());
    return dist(rng);
}

std::vector<int32_t> sample_cpu_rows_general(const Tensor &logits, const std::vector<int32_t> &sample_rows,
                                             int32_t vocab_size,
                                             const std::vector<std::vector<int32_t>> *token_histories,
                                             const std::vector<SamplingParams> *sampling_params,
                                             const std::vector<uint64_t> *request_ids)
{
    std::vector<int32_t> sampled(static_cast<size_t>(logits.size(0)), -1);
    Tensor logits_cpu = tensor_to_cpu_contiguous(logits);
    const float *logits_ptr = logits_cpu.data_ptr<float>();
    const SamplingParams default_params;
    const std::vector<int32_t> empty_history;

    for (size_t sample_index = 0; sample_index < sample_rows.size(); ++sample_index)
    {
        const int32_t row = sample_rows[sample_index];
        const float *row_logits = logits_ptr + static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
        std::vector<float> row_values(row_logits, row_logits + vocab_size);
        const SamplingParams &params = sampling_params == nullptr ? default_params : (*sampling_params)[sample_index];
        const std::vector<int32_t> &history =
            token_histories == nullptr ? empty_history : (*token_histories)[sample_index];
        const uint64_t request_id =
            request_ids == nullptr ? static_cast<uint64_t>(sample_index) : (*request_ids)[sample_index];
        sampled[static_cast<size_t>(row)] =
            sample_filtered_row(std::move(row_values), history, vocab_size, params, request_id, sample_index);
    }
    return sampled;
}

std::vector<int32_t> sample_cuda_rows_without_repetition_penalty(const Tensor &logits,
                                                                 const std::vector<int32_t> &sample_rows)
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
        Tensor row_tensor =
            torch::tensor(row_indices, torch::TensorOptions().dtype(torch::kInt64).device(logits.device()));
        sampled_tokens = logits.index_select(0, row_tensor).argmax(/*dim=*/1);
    }

    Tensor sampled_cpu = sampled_tokens
                             .to(torch::TensorOptions().dtype(torch::kInt32).device(c10::kCPU),
                                 /*non_blocking=*/false,
                                 /*copy=*/true)
                             .contiguous();
    const int32_t *token_ptr = sampled_cpu.data_ptr<int32_t>();
    return std::vector<int32_t>(token_ptr, token_ptr + sample_rows.size());
}

std::vector<int32_t> sample_cuda_rows_with_repetition_penalty(const Tensor &logits,
                                                              const std::vector<int32_t> &sample_rows,
                                                              int32_t vocab_size,
                                                              const std::vector<std::vector<int32_t>> &token_histories,
                                                              const std::vector<SamplingParams> &sampling_params)
{
#if TINYLLM_ENABLE_CUDA
    const int32_t sample_count = static_cast<int32_t>(sample_rows.size());
    std::vector<int32_t> history_offsets;
    std::vector<int64_t> history_tokens;
    std::vector<float> penalties;
    history_offsets.reserve(sample_rows.size() + 1);
    penalties.reserve(sample_rows.size());
    history_offsets.push_back(0);
    for (size_t sample_index = 0; sample_index < sample_rows.size(); ++sample_index)
    {
        const float penalty = sampling_params[sample_index].repetition_penalty;
        if (penalty <= 0.0f)
        {
            throw std::runtime_error("sample_rows: repetition penalty must be positive.");
        }
        penalties.push_back(penalty);
        if (penalty != 1.0f && !token_histories[sample_index].empty())
        {
            const std::vector<int64_t> unique_tokens =
                unique_valid_history_tokens(token_histories[sample_index], vocab_size);
            history_tokens.insert(history_tokens.end(), unique_tokens.begin(), unique_tokens.end());
        }
        history_offsets.push_back(static_cast<int32_t>(history_tokens.size()));
    }

    const auto device = logits.device();
    CudaSamplerScratch &scratch =
        get_cuda_sampler_scratch(device, sample_count, vocab_size, static_cast<int32_t>(history_tokens.size()));
    Tensor sample_rows_tensor = scratch.sample_rows.narrow(0, 0, sample_count);
    Tensor history_offsets_tensor = scratch.history_offsets.narrow(0, 0, sample_count + 1);
    Tensor penalties_tensor = scratch.penalties.narrow(0, 0, sample_count);
    Tensor history_mask = scratch.history_mask.narrow(0, 0, sample_count);
    Tensor sampled_tokens = scratch.sampled_tokens.narrow(0, 0, sample_count);

    sample_rows_tensor.copy_(vector_cpu_view(sample_rows, torch::kInt32), false);
    history_offsets_tensor.copy_(vector_cpu_view(history_offsets, torch::kInt32), false);
    penalties_tensor.copy_(vector_cpu_view(penalties, torch::kFloat32), false);
    history_mask.zero_();
    if (!history_tokens.empty())
    {
        Tensor history_tokens_tensor = scratch.history_tokens.narrow(0, 0, static_cast<int64_t>(history_tokens.size()));
        history_tokens_tensor.copy_(vector_cpu_view(history_tokens, torch::kInt64), false);
        cuda::launch_mark_repetition_history_mask(
            history_tokens_tensor.data_ptr<int64_t>(), history_offsets_tensor.data_ptr<int32_t>(),
            history_mask.data_ptr<uint8_t>(), sample_count, vocab_size, at::cuda::getCurrentCUDAStream(device.index()));
    }

    cuda::launch_argmax_repetition_penalty_f32(logits.data_ptr<float>(), sample_rows_tensor.data_ptr<int32_t>(),
                                               history_mask.data_ptr<uint8_t>(), penalties_tensor.data_ptr<float>(),
                                               sampled_tokens.data_ptr<int32_t>(), sample_count, vocab_size,
                                               at::cuda::getCurrentCUDAStream(device.index()));

    Tensor sampled_cpu = sampled_tokens
                             .to(torch::TensorOptions().dtype(torch::kInt32).device(c10::kCPU),
                                 /*non_blocking=*/false,
                                 /*copy=*/true)
                             .contiguous();
    const int32_t *token_ptr = sampled_cpu.data_ptr<int32_t>();
    return std::vector<int32_t>(token_ptr, token_ptr + sample_rows.size());
#else
    std::vector<Tensor> sampled_per_row;
    sampled_per_row.reserve(sample_rows.size());
    const auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(logits.device());
    for (size_t sample_index = 0; sample_index < sample_rows.size(); ++sample_index)
    {
        const int32_t row = sample_rows[sample_index];
        const float penalty = sampling_params[sample_index].repetition_penalty;
        if (penalty <= 0.0f)
        {
            throw std::runtime_error("sample_rows: repetition penalty must be positive.");
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
                Tensor adjusted = torch::where(history_logits > 0.0f, history_logits / static_cast<double>(penalty),
                                               history_logits * static_cast<double>(penalty));
                row_logits.index_put_({token_indices}, adjusted);
            }
        }
        sampled_per_row.push_back(row_logits.argmax(/*dim=*/0).reshape({1}));
    }

    Tensor sampled_tokens = torch::cat(sampled_per_row, /*dim=*/0);
    Tensor sampled_cpu = sampled_tokens
                             .to(torch::TensorOptions().dtype(torch::kInt32).device(c10::kCPU),
                                 /*non_blocking=*/false,
                                 /*copy=*/true)
                             .contiguous();
    const int32_t *token_ptr = sampled_cpu.data_ptr<int32_t>();
    return std::vector<int32_t>(token_ptr, token_ptr + sample_rows.size());
#endif
}

int32_t sample_argmax(const float *logits, int32_t vocab_size)
{
    if (logits == nullptr)
    {
        throw std::runtime_error("sample_rows: logits pointer must be non-null.");
    }
    if (vocab_size <= 0)
    {
        throw std::runtime_error("sample_rows: vocab_size must be positive.");
    }

    int32_t best_token = 0;
    float best_value = logits[0];
    for (int32_t token = 1; token < vocab_size; ++token)
    {
        if (logits[token] > best_value)
        {
            best_value = logits[token];
            best_token = token;
        }
    }
    return best_token;
}

int32_t sample_argmax_with_repetition_penalty(const float *logits, int32_t vocab_size,
                                              const std::vector<int32_t> &history, float penalty)
{
    std::unordered_set<int32_t> seen_tokens;
    seen_tokens.reserve(history.size());
    for (int32_t token_id : history)
    {
        if (token_id < 0 || token_id >= vocab_size)
        {
            throw std::runtime_error("sample_rows: token history id is out of vocabulary range.");
        }
        seen_tokens.insert(token_id);
    }

    int32_t best_token = 0;
    float best_value =
        seen_tokens.find(0) == seen_tokens.end() ? logits[0] : apply_repetition_penalty_to_logit(logits[0], penalty);
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

} // namespace

static std::vector<int32_t> sample_rows_impl(const Tensor &logits, const std::vector<int32_t> &sample_rows,
                                             int32_t vocab_size,
                                             const std::vector<std::vector<int32_t>> *token_histories,
                                             const std::vector<SamplingParams> *sampling_params,
                                             const std::vector<uint64_t> *request_ids)
{
    if (!logits.defined())
    {
        throw std::runtime_error("sample_rows: logits must be defined.");
    }
    if (tensor_dtype(logits) != DType::kFloat32)
    {
        throw std::runtime_error("sample_rows: logits must be float32.");
    }
    if (vocab_size <= 0)
    {
        throw std::runtime_error("sample_rows: vocab_size must be positive.");
    }
    if (logits.dim() != 2 || logits.size(1) != vocab_size)
    {
        throw std::runtime_error("sample_rows: logits shape must be [rows, vocab_size].");
    }

    std::vector<int32_t> sampled(static_cast<size_t>(logits.size(0)), -1);
    for (int32_t row : sample_rows)
    {
        if (row < 0 || row >= logits.size(0))
        {
            throw std::runtime_error("sample_rows: sample row is out of range.");
        }
    }

    if (sample_rows.empty())
    {
        return sampled;
    }

    validate_sampling_metadata(sample_rows, token_histories, sampling_params, request_ids);
    if (has_non_greedy_sampling(sample_rows, sampling_params))
    {
        return sample_cpu_rows_general(logits, sample_rows, vocab_size, token_histories, sampling_params, request_ids);
    }

    const bool apply_repetition_penalty = has_active_repetition_penalty(sample_rows, token_histories, sampling_params);
    if (logits.device().is_cuda())
    {
        const std::vector<int32_t> sampled_tokens =
            apply_repetition_penalty ? sample_cuda_rows_with_repetition_penalty(logits, sample_rows, vocab_size,
                                                                                *token_histories, *sampling_params)
                                     : sample_cuda_rows_without_repetition_penalty(logits, sample_rows);
        for (size_t i = 0; i < sample_rows.size(); ++i)
        {
            sampled[static_cast<size_t>(sample_rows[i])] = sampled_tokens[i];
        }
        return sampled;
    }

    Tensor logits_cpu = tensor_to_cpu_contiguous(logits);
    const float *logits_ptr = logits_cpu.data_ptr<float>();
    for (size_t sample_index = 0; sample_index < sample_rows.size(); ++sample_index)
    {
        const int32_t row = sample_rows[sample_index];
        const float *row_logits = logits_ptr + static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
        if (!apply_repetition_penalty || (*sampling_params)[sample_index].repetition_penalty == 1.0f)
        {
            sampled[static_cast<size_t>(row)] = sample_argmax(row_logits, vocab_size);
            continue;
        }

        sampled[static_cast<size_t>(row)] =
            sample_argmax_with_repetition_penalty(row_logits, vocab_size, (*token_histories)[sample_index],
                                                  (*sampling_params)[sample_index].repetition_penalty);
    }
    return sampled;
}

std::vector<int32_t> sample_rows(const Tensor &logits, const SamplerBatch &batch)
{
    return sample_rows_impl(logits, batch.sample_rows, batch.vocab_size, batch.token_histories, batch.sampling_params,
                            batch.request_ids);
}

} // namespace tiny_llm
