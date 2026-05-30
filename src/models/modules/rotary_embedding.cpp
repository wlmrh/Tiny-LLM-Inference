#include "tiny_llm/models/modules/rotary_embedding.h"

#include "tiny_llm/operators/llama_ops.h"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <mutex>
#include <vector>
#include <utility>

namespace tiny_llm {
namespace modules {

namespace {

struct RopeCosSinCacheEntry {
    c10::Device device{c10::kCPU};
    int32_t head_dim = 0;
    float rope_theta = 0.0f;
    std::string rope_scaling_type;
    float rope_scaling_factor = 1.0f;
    float rope_scaling_low_freq_factor = 1.0f;
    float rope_scaling_high_freq_factor = 1.0f;
    int32_t rope_scaling_original_max_position_embeddings = 0;
    int32_t max_positions = 0;
    Tensor cos;
    Tensor sin;
};

std::mutex g_rope_cos_sin_cache_mutex;
std::vector<RopeCosSinCacheEntry> g_rope_cos_sin_cache;

int32_t rope_cache_length(int32_t max_position_embeddings, int32_t rope_scaling_original_max_position_embeddings)
{
    return std::max(4096, std::max(max_position_embeddings, rope_scaling_original_max_position_embeddings));
}

bool cache_entry_matches(const RopeCosSinCacheEntry& entry,
                         const c10::Device& device,
                         int32_t head_dim,
                         float rope_theta,
                         const std::string& rope_scaling_type,
                         float rope_scaling_factor,
                         float rope_scaling_low_freq_factor,
                         float rope_scaling_high_freq_factor,
                         int32_t rope_scaling_original_max_position_embeddings,
                         int32_t required_positions)
{
    return entry.cos.defined()
        && entry.sin.defined()
        && entry.device == device
        && entry.head_dim == head_dim
        && entry.rope_theta == rope_theta
        && entry.rope_scaling_type == rope_scaling_type
        && entry.rope_scaling_factor == rope_scaling_factor
        && entry.rope_scaling_low_freq_factor == rope_scaling_low_freq_factor
        && entry.rope_scaling_high_freq_factor == rope_scaling_high_freq_factor
        && entry.rope_scaling_original_max_position_embeddings == rope_scaling_original_max_position_embeddings
        && entry.max_positions >= required_positions;
}

} // namespace

RotaryEmbedding::RotaryEmbedding(int32_t num_attention_heads,
                                 int32_t num_key_value_heads,
                                 int32_t head_dim,
                                 float rope_theta,
                                 std::string rope_scaling_type,
                                 float rope_scaling_factor,
                                 float rope_scaling_low_freq_factor,
                                 float rope_scaling_high_freq_factor,
                                 int32_t rope_scaling_original_max_position_embeddings,
                                 int32_t max_position_embeddings)
    : num_attention_heads_(num_attention_heads),
      num_key_value_heads_(num_key_value_heads),
      head_dim_(head_dim),
      rope_theta_(rope_theta),
      rope_scaling_type_(std::move(rope_scaling_type)),
      rope_scaling_factor_(rope_scaling_factor),
      rope_scaling_low_freq_factor_(rope_scaling_low_freq_factor),
      rope_scaling_high_freq_factor_(rope_scaling_high_freq_factor),
      rope_scaling_original_max_position_embeddings_(rope_scaling_original_max_position_embeddings),
      max_position_embeddings_(max_position_embeddings)
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

std::pair<Tensor, Tensor> RotaryEmbedding::cos_sin_for_device(const c10::Device& device) const
{
    const int32_t required_positions = rope_cache_length(
        max_position_embeddings_,
        rope_scaling_original_max_position_embeddings_);

    {
        std::lock_guard<std::mutex> lock(g_rope_cos_sin_cache_mutex);
        for (const RopeCosSinCacheEntry& entry : g_rope_cos_sin_cache)
        {
            if (cache_entry_matches(
                    entry,
                    device,
                    head_dim_,
                    rope_theta_,
                    rope_scaling_type_,
                    rope_scaling_factor_,
                    rope_scaling_low_freq_factor_,
                    rope_scaling_high_freq_factor_,
                    rope_scaling_original_max_position_embeddings_,
                    required_positions))
            {
                return {entry.cos, entry.sin};
            }
        }
    }

    const Tensor inv_freq = inv_freq_for_device(device);
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const Tensor positions = torch::arange(required_positions, options).unsqueeze(1);
    const Tensor theta = positions * inv_freq.unsqueeze(0);
    RopeCosSinCacheEntry entry;
    entry.device = device;
    entry.head_dim = head_dim_;
    entry.rope_theta = rope_theta_;
    entry.rope_scaling_type = rope_scaling_type_;
    entry.rope_scaling_factor = rope_scaling_factor_;
    entry.rope_scaling_low_freq_factor = rope_scaling_low_freq_factor_;
    entry.rope_scaling_high_freq_factor = rope_scaling_high_freq_factor_;
    entry.rope_scaling_original_max_position_embeddings = rope_scaling_original_max_position_embeddings_;
    entry.max_positions = required_positions;
    entry.cos = torch::cos(theta).contiguous();
    entry.sin = torch::sin(theta).contiguous();

    std::lock_guard<std::mutex> lock(g_rope_cos_sin_cache_mutex);
    g_rope_cos_sin_cache.push_back(entry);
    return {entry.cos, entry.sin};
}

void RotaryEmbedding::forward(const Tensor& positions, Tensor& q, Tensor& k) const
{
    if (q.device().is_cuda())
    {
        const auto cache = cos_sin_for_device(q.device());
        ops::apply_rope(
            positions,
            q,
            k,
            num_attention_heads_,
            num_key_value_heads_,
            head_dim_,
            cache.first,
            cache.second);
        return;
    }

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
