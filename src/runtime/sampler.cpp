#include "tiny_llm/runtime/sampler.h"

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/sampling.h"

#include <stdexcept>

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

    Tensor logits_cpu = tensor_to_cpu_contiguous(logits);
    const float* logits_ptr = logits_cpu.data_ptr<float>();
    std::vector<int32_t> sampled(static_cast<size_t>(logits.size(0)), -1);
    for (int32_t row : sample_rows)
    {
        if (row < 0 || row >= logits.size(0))
        {
            throw std::runtime_error("sample_greedy_rows: sample row is out of range.");
        }
        sampled[static_cast<size_t>(row)] = sample_argmax(
            logits_ptr + static_cast<size_t>(row) * static_cast<size_t>(vocab_size),
            vocab_size);
    }
    return sampled;
}

} // namespace tiny_llm
