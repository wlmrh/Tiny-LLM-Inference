#include "tiny_llm/models/modules/rotary_embedding.h"

#include "tiny_llm/operators/llama_ops.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace tiny_llm {
namespace modules {

RotaryEmbedding::RotaryEmbedding(int32_t num_attention_heads,
                                 int32_t num_key_value_heads,
                                 int32_t head_dim,
                                 float rope_theta,
                                 std::string rope_scaling_type,
                                 float rope_scaling_factor,
                                 float rope_scaling_low_freq_factor,
                                 float rope_scaling_high_freq_factor,
                                 int32_t rope_scaling_original_max_position_embeddings)
    : num_attention_heads_(num_attention_heads),
      num_key_value_heads_(num_key_value_heads),
      head_dim_(head_dim),
      rope_theta_(rope_theta),
      rope_scaling_type_(std::move(rope_scaling_type)),
      rope_scaling_factor_(rope_scaling_factor),
      rope_scaling_low_freq_factor_(rope_scaling_low_freq_factor),
      rope_scaling_high_freq_factor_(rope_scaling_high_freq_factor),
      rope_scaling_original_max_position_embeddings_(rope_scaling_original_max_position_embeddings)
{
    if (num_attention_heads_ <= 0 || num_key_value_heads_ <= 0 || head_dim_ <= 0 || rope_theta_ <= 0.0f
        || rope_scaling_factor_ <= 0.0f)
    {
        throw std::runtime_error("modules::RotaryEmbedding: invalid configuration.");
    }
}

Tensor RotaryEmbedding::inv_freq_for_device(const c10::Device& device) const
{
    if (cached_inv_freq_.defined() && cached_inv_freq_.device() == device)
    {
        return cached_inv_freq_;
    }

    const int64_t rotary_half = head_dim_ / 2;
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const Tensor dim = torch::arange(rotary_half, options);
    const Tensor exponent = dim * (2.0f / static_cast<float>(head_dim_));
    const Tensor base = torch::full({rotary_half}, rope_theta_, options);
    Tensor inv_freq = 1.0f / torch::pow(base, exponent);
    if (rope_scaling_type_ == "llama3")
    {
        const float low_freq_wavelen =
            static_cast<float>(rope_scaling_original_max_position_embeddings_) / rope_scaling_low_freq_factor_;
        const float high_freq_wavelen =
            static_cast<float>(rope_scaling_original_max_position_embeddings_) / rope_scaling_high_freq_factor_;
        const Tensor wavelen = (2.0f * static_cast<float>(M_PI)) / inv_freq;
        const Tensor smooth_factor =
            (static_cast<float>(rope_scaling_original_max_position_embeddings_) / wavelen
             - rope_scaling_low_freq_factor_)
            / (rope_scaling_high_freq_factor_ - rope_scaling_low_freq_factor_);
        const Tensor medium_freq =
            (1.0f - smooth_factor) * (inv_freq / rope_scaling_factor_) + smooth_factor * inv_freq;
        inv_freq = torch::where(
            wavelen > low_freq_wavelen,
            inv_freq / rope_scaling_factor_,
            torch::where(wavelen < high_freq_wavelen, inv_freq, medium_freq));
    }
    else if (!rope_scaling_type_.empty())
    {
        inv_freq = inv_freq / rope_scaling_factor_;
    }

    cached_inv_freq_ = inv_freq.contiguous();
    return cached_inv_freq_;
}

void RotaryEmbedding::forward(const Tensor& positions, Tensor& q, Tensor& k) const
{
    ops::apply_rope(
        positions,
        q,
        k,
        num_attention_heads_,
        num_key_value_heads_,
        head_dim_,
        inv_freq_for_device(q.device()));
}

} // namespace modules
} // namespace tiny_llm
