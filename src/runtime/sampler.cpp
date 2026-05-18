#include "tiny_llm/runtime/sampler.h"

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/sampling.h"

#include <stdexcept>
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

} // namespace

std::vector<int32_t> sample_greedy_rows(const Tensor& logits,
                                        const std::vector<int32_t>& sample_rows,
                                        int32_t vocab_size)
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

    if (logits.device().is_cuda())
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
        Tensor sampled_tokens = logits.index_select(0, row_tensor).argmax(/*dim=*/1);
        Tensor sampled_cpu = sampled_tokens.to(
            torch::TensorOptions().dtype(torch::kInt32).device(c10::kCPU),
            /*non_blocking=*/false,
            /*copy=*/true).contiguous();
        const int32_t* token_ptr = sampled_cpu.data_ptr<int32_t>();
        for (size_t i = 0; i < sample_rows.size(); ++i)
        {
            sampled[static_cast<size_t>(sample_rows[i])] = token_ptr[i];
        }
        return sampled;
    }

    Tensor logits_cpu = tensor_to_cpu_contiguous(logits);
    const float* logits_ptr = logits_cpu.data_ptr<float>();
    for (int32_t row : sample_rows)
    {
        sampled[static_cast<size_t>(row)] = sample_argmax(
            logits_ptr + static_cast<size_t>(row) * static_cast<size_t>(vocab_size),
            vocab_size);
    }
    return sampled;
}

} // namespace tiny_llm
