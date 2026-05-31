#include "tiny_llm/operators/llama_ops.h"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

#if TINYLLM_ENABLE_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

namespace tiny_llm {
namespace ops {

#if TINYLLM_ENABLE_CUDA
namespace cuda {
void launch_apply_rope_cached_f32(const int32_t* positions,
                                  float* q,
                                  float* k,
                                  const float* cos_cache,
                                  const float* sin_cache,
                                  int64_t rows,
                                  int32_t num_attention_heads,
                                  int32_t num_key_value_heads,
                                  int32_t head_dim,
                                  int64_t cache_rows,
                                  int64_t q_stride,
                                  int64_t k_stride,
                                  cudaStream_t stream);
void launch_silu_multiply_f32(const float* gate,
                              const float* up,
                              float* out,
                              int64_t numel,
                              cudaStream_t stream);
void launch_add_f32(const float* lhs,
                    const float* rhs,
                    float* out,
                    int64_t numel,
                    cudaStream_t stream);
} // namespace cuda
#endif

namespace {

bool any_cuda(std::initializer_list<std::reference_wrapper<const Tensor>> tensors)
{
    for (const Tensor& tensor : tensors)
    {
        if (tensor.defined() && tensor.device().is_cuda())
        {
            return true;
        }
    }
    return false;
}

void validate_same_device(std::initializer_list<std::reference_wrapper<const Tensor>> tensors, const char* name)
{
    bool have_device = false;
    c10::Device device(c10::kCPU);
    for (const Tensor& tensor : tensors)
    {
        if (!tensor.defined())
        {
            throw std::runtime_error(std::string(name) + ": tensor must be defined.");
        }
        if (!have_device)
        {
            device = tensor.device();
            have_device = true;
            continue;
        }
        if (tensor.device() != device)
        {
            throw std::runtime_error(std::string(name) + ": tensors must be on the same device.");
        }
    }
}

void validate_cpu_tensor(const Tensor& tensor, const char* name)
{
    if (tensor.device().is_cuda())
    {
        throw std::runtime_error(std::string(name) + ": CUDA path is not implemented.");
    }
}

void validate_int_tensor_1d(const Tensor& tensor, int64_t size, const char* name)
{
    if (!tensor.defined())
    {
        throw std::runtime_error(std::string(name) + ": tensor must be defined.");
    }
    if (tensor_dtype(tensor) != DType::kInt32)
    {
        throw std::runtime_error(std::string(name) + ": tensor must be int32.");
    }
    if (tensor.dim() != 1 || tensor.size(0) != size)
    {
        throw std::runtime_error(std::string(name) + ": unexpected shape.");
    }
    if (tensor_data(tensor) == nullptr)
    {
        throw std::runtime_error(std::string(name) + ": tensor data pointer must be non-null.");
    }
}

void validate_float_tensor_1d(const Tensor& tensor, int64_t size, const char* name)
{
    if (!tensor.defined())
    {
        throw std::runtime_error(std::string(name) + ": tensor must be defined.");
    }
    if (tensor_dtype(tensor) != DType::kFloat32)
    {
        throw std::runtime_error(std::string(name) + ": tensor must be float32.");
    }
    if (tensor.dim() != 1 || tensor.size(0) != size)
    {
        throw std::runtime_error(std::string(name) + ": unexpected shape.");
    }
    if (tensor_data(tensor) == nullptr)
    {
        throw std::runtime_error(std::string(name) + ": tensor data pointer must be non-null.");
    }
}

void validate_float_tensor_2d(const Tensor& tensor,
                              int64_t rows,
                              int64_t cols,
                              const char* name)
{
    if (!tensor.defined())
    {
        throw std::runtime_error(std::string(name) + ": tensor must be defined.");
    }
    if (tensor_dtype(tensor) != DType::kFloat32)
    {
        throw std::runtime_error(std::string(name) + ": tensor must be float32.");
    }
    if (tensor.dim() != 2 || tensor.size(0) != rows || tensor.size(1) != cols)
    {
        throw std::runtime_error(std::string(name) + ": unexpected shape.");
    }
    if (tensor_data(tensor) == nullptr)
    {
        throw std::runtime_error(std::string(name) + ": tensor data pointer must be non-null.");
    }
}

void run_embedding_lookup_device(const Tensor& ids,
                                 const Tensor& embedding,
                                 Tensor& out,
                                 bool embedding_is_vocab_hidden)
{
    validate_same_device({std::cref(ids), std::cref(embedding), std::cref(out)}, "embedding_lookup");
    const Tensor ids_long = ids.to(torch::TensorOptions().dtype(torch::kInt64).device(ids.device()));
    const Tensor embedding_vocab_hidden = embedding_is_vocab_hidden
        ? embedding
        : embedding.transpose(0, 1);
    out.copy_(embedding_vocab_hidden.index_select(0, ids_long));
}

void run_apply_rope_device_with_inv_freq(const Tensor& positions,
                                        Tensor& q,
                                        Tensor& k,
                                        int32_t num_attention_heads,
                                        int32_t num_key_value_heads,
                                        int32_t head_dim,
                                        const Tensor& inv_freq)
{
    validate_same_device({std::cref(positions), std::cref(q), std::cref(k), std::cref(inv_freq)}, "apply_rope");
    const int64_t rows = q.size(0);
    const int64_t rotary_half = head_dim / 2;
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    const Tensor positions_f = positions.to(options);
    const Tensor theta = positions_f.unsqueeze(1) * inv_freq;
    const Tensor cos_theta = torch::cos(theta).unsqueeze(1);
    const Tensor sin_theta = torch::sin(theta).unsqueeze(1);

    Tensor q_view = q.view({rows, num_attention_heads, head_dim});
    Tensor q_first = q_view.narrow(2, 0, rotary_half);
    Tensor q_second = q_view.narrow(2, rotary_half, rotary_half);
    Tensor q_first_old = q_first.clone();
    Tensor q_second_old = q_second.clone();
    q_first.copy_(q_first_old * cos_theta - q_second_old * sin_theta);
    q_second.copy_(q_second_old * cos_theta + q_first_old * sin_theta);

    Tensor k_view = k.view({rows, num_key_value_heads, head_dim});
    Tensor k_first = k_view.narrow(2, 0, rotary_half);
    Tensor k_second = k_view.narrow(2, rotary_half, rotary_half);
    Tensor k_first_old = k_first.clone();
    Tensor k_second_old = k_second.clone();
    k_first.copy_(k_first_old * cos_theta - k_second_old * sin_theta);
    k_second.copy_(k_second_old * cos_theta + k_first_old * sin_theta);
}

void run_apply_rope_device_with_cache(const Tensor& positions,
                                      Tensor& q,
                                      Tensor& k,
                                      int32_t num_attention_heads,
                                      int32_t num_key_value_heads,
                                      int32_t head_dim,
                                      const Tensor& cos_cache,
                                      const Tensor& sin_cache)
{
    validate_same_device({std::cref(positions), std::cref(q), std::cref(k), std::cref(cos_cache), std::cref(sin_cache)}, "apply_rope");
#if TINYLLM_ENABLE_CUDA
    if (positions.device().is_cuda()
        && q.device().is_cuda()
        && k.device().is_cuda()
        && cos_cache.device().is_cuda()
        && sin_cache.device().is_cuda()
        && positions.is_contiguous()
        && q.is_contiguous()
        && k.is_contiguous()
        && cos_cache.is_contiguous()
        && sin_cache.is_contiguous())
    {
        cuda::launch_apply_rope_cached_f32(
            positions.data_ptr<int32_t>(),
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            cos_cache.data_ptr<float>(),
            sin_cache.data_ptr<float>(),
            q.size(0),
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            cos_cache.size(0),
            q.stride(0),
            k.stride(0),
            at::cuda::getCurrentCUDAStream(q.device().index()));
        return;
    }
#endif
    const int64_t rows = q.size(0);
    const int64_t rotary_half = head_dim / 2;
    const Tensor positions_long = positions.to(torch::TensorOptions().dtype(torch::kInt64).device(positions.device()));
    const Tensor cos_theta = cos_cache.index_select(0, positions_long).unsqueeze(1);
    const Tensor sin_theta = sin_cache.index_select(0, positions_long).unsqueeze(1);

    Tensor q_view = q.view({rows, num_attention_heads, head_dim});
    Tensor q_first = q_view.narrow(2, 0, rotary_half);
    Tensor q_second = q_view.narrow(2, rotary_half, rotary_half);
    Tensor q_first_old = q_first.clone();
    Tensor q_second_old = q_second.clone();
    q_first.copy_(q_first_old * cos_theta - q_second_old * sin_theta);
    q_second.copy_(q_second_old * cos_theta + q_first_old * sin_theta);

    Tensor k_view = k.view({rows, num_key_value_heads, head_dim});
    Tensor k_first = k_view.narrow(2, 0, rotary_half);
    Tensor k_second = k_view.narrow(2, rotary_half, rotary_half);
    Tensor k_first_old = k_first.clone();
    Tensor k_second_old = k_second.clone();
    k_first.copy_(k_first_old * cos_theta - k_second_old * sin_theta);
    k_second.copy_(k_second_old * cos_theta + k_first_old * sin_theta);
}

void run_apply_rope_device(const Tensor& positions,
                           Tensor& q,
                           Tensor& k,
                           int32_t num_attention_heads,
                           int32_t num_key_value_heads,
                           int32_t head_dim,
                           float rope_theta,
                           const std::string& rope_scaling_type,
                           float rope_scaling_factor,
                           float rope_scaling_low_freq_factor,
                           float rope_scaling_high_freq_factor,
                           int32_t rope_scaling_original_max_position_embeddings)
{
    validate_same_device({std::cref(positions), std::cref(q), std::cref(k)}, "apply_rope");
    const int64_t rotary_half = head_dim / 2;
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    const Tensor dim = torch::arange(rotary_half, options);
    const Tensor exponent = dim * (2.0f / static_cast<float>(head_dim));
    const Tensor base = torch::full({rotary_half}, rope_theta, options);
    Tensor inv_freq = 1.0f / torch::pow(base, exponent);
    if (rope_scaling_type == "llama3")
    {
        const float low_freq_wavelen =
            static_cast<float>(rope_scaling_original_max_position_embeddings) / rope_scaling_low_freq_factor;
        const float high_freq_wavelen =
            static_cast<float>(rope_scaling_original_max_position_embeddings) / rope_scaling_high_freq_factor;
        const Tensor wavelen = (2.0f * static_cast<float>(M_PI)) / inv_freq;
        const Tensor smooth_factor =
            (static_cast<float>(rope_scaling_original_max_position_embeddings) / wavelen
             - rope_scaling_low_freq_factor)
            / (rope_scaling_high_freq_factor - rope_scaling_low_freq_factor);
        const Tensor medium_freq =
            (1.0f - smooth_factor) * (inv_freq / rope_scaling_factor) + smooth_factor * inv_freq;
        inv_freq = torch::where(
            wavelen > low_freq_wavelen,
            inv_freq / rope_scaling_factor,
            torch::where(wavelen < high_freq_wavelen, inv_freq, medium_freq));
    }
    else if (!rope_scaling_type.empty())
    {
        inv_freq = inv_freq / rope_scaling_factor;
    }
    run_apply_rope_device_with_inv_freq(
        positions,
        q,
        k,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        inv_freq);
}

} // namespace

void embedding_lookup(const Tensor& ids,
                      const Tensor& embedding,
                      Tensor& out,
                      int32_t vocab_size,
                      int32_t hidden_size,
                      bool embedding_is_vocab_hidden)
{
    if (tensor_dtype(ids) != DType::kInt32 || tensor_dtype(embedding) != DType::kFloat32
        || tensor_dtype(out) != DType::kFloat32)
    {
        throw std::runtime_error("embedding_lookup: dtype mismatch.");
    }
    if (ids.dim() != 1 || out.dim() != 2 || out.size(0) != ids.size(0) || out.size(1) != hidden_size)
    {
        throw std::runtime_error("embedding_lookup: ids/out shape mismatch.");
    }
    if (tensor_data(ids) == nullptr || tensor_data(embedding) == nullptr || tensor_data(out) == nullptr)
    {
        throw std::runtime_error("embedding_lookup: data pointers must be non-null.");
    }
    if (any_cuda({std::cref(ids), std::cref(embedding), std::cref(out)}))
    {
        run_embedding_lookup_device(ids, embedding, out, embedding_is_vocab_hidden);
        return;
    }
    validate_cpu_tensor(ids, "embedding_lookup");
    validate_cpu_tensor(embedding, "embedding_lookup");
    validate_cpu_tensor(out, "embedding_lookup");

    const int32_t* ids_ptr = static_cast<const int32_t*>(tensor_data(ids));
    const float* embed_ptr = static_cast<const float*>(tensor_data(embedding));
    float* out_ptr = static_cast<float*>(tensor_data(out));

    const int64_t stride0 = embedding.stride(0);
    const int64_t stride1 = embedding.stride(1);
    const int64_t rows = ids.size(0);
    for (int64_t row = 0; row < rows; ++row)
    {
        const int32_t token_id = ids_ptr[row];
        if (token_id < 0 || token_id >= vocab_size)
        {
            throw std::runtime_error("embedding_lookup: token id is out of range.");
        }

        float* out_row = out_ptr + static_cast<size_t>(row) * static_cast<size_t>(hidden_size);
        for (int32_t col = 0; col < hidden_size; ++col)
        {
            if (embedding_is_vocab_hidden)
            {
                out_row[static_cast<size_t>(col)] = embed_ptr[
                    static_cast<size_t>(token_id) * static_cast<size_t>(stride0)
                    + static_cast<size_t>(col) * static_cast<size_t>(stride1)];
            }
            else
            {
                out_row[static_cast<size_t>(col)] = embed_ptr[
                    static_cast<size_t>(col) * static_cast<size_t>(stride0)
                    + static_cast<size_t>(token_id) * static_cast<size_t>(stride1)];
            }
        }
    }
}

void split_qkv(const Tensor& qkv,
               Tensor& q,
               Tensor& k,
               Tensor& v,
               int32_t hidden_size,
               int32_t kv_hidden_size)
{
    const int64_t rows = qkv.size(0);
    validate_float_tensor_2d(qkv, rows, hidden_size + 2 * kv_hidden_size, "split_qkv::qkv");
    validate_float_tensor_2d(q, rows, hidden_size, "split_qkv::q");
    validate_float_tensor_2d(k, rows, kv_hidden_size, "split_qkv::k");
    validate_float_tensor_2d(v, rows, kv_hidden_size, "split_qkv::v");

    if (any_cuda({std::cref(qkv), std::cref(q), std::cref(k), std::cref(v)}))
    {
        validate_same_device({std::cref(qkv), std::cref(q), std::cref(k), std::cref(v)}, "split_qkv");
        q.copy_(qkv.narrow(1, 0, hidden_size));
        k.copy_(qkv.narrow(1, hidden_size, kv_hidden_size));
        v.copy_(qkv.narrow(1, hidden_size + kv_hidden_size, kv_hidden_size));
        return;
    }

    validate_cpu_tensor(qkv, "split_qkv");
    validate_cpu_tensor(q, "split_qkv");
    validate_cpu_tensor(k, "split_qkv");
    validate_cpu_tensor(v, "split_qkv");

    const float* qkv_ptr = static_cast<const float*>(tensor_data(qkv));
    float* q_ptr = static_cast<float*>(tensor_data(q));
    float* k_ptr = static_cast<float*>(tensor_data(k));
    float* v_ptr = static_cast<float*>(tensor_data(v));

    for (int64_t row = 0; row < rows; ++row)
    {
        const size_t qkv_offset =
            static_cast<size_t>(row) * static_cast<size_t>(hidden_size + 2 * kv_hidden_size);
        const size_t out_offset = static_cast<size_t>(row) * static_cast<size_t>(hidden_size);
        const size_t kv_out_offset = static_cast<size_t>(row) * static_cast<size_t>(kv_hidden_size);
        for (int32_t col = 0; col < hidden_size; ++col)
        {
            q_ptr[out_offset + static_cast<size_t>(col)] = qkv_ptr[qkv_offset + static_cast<size_t>(col)];
        }
        for (int32_t col = 0; col < kv_hidden_size; ++col)
        {
            k_ptr[kv_out_offset + static_cast<size_t>(col)] =
                qkv_ptr[qkv_offset + static_cast<size_t>(hidden_size + col)];
            v_ptr[kv_out_offset + static_cast<size_t>(col)] =
                qkv_ptr[qkv_offset + static_cast<size_t>(hidden_size + kv_hidden_size + col)];
        }
    }
}

void apply_rope(const Tensor& positions,
                Tensor& q,
                Tensor& k,
                int32_t num_attention_heads,
                int32_t num_key_value_heads,
                int32_t head_dim,
                float rope_theta,
                const char* rope_scaling_type,
                float rope_scaling_factor,
                float rope_scaling_low_freq_factor,
                float rope_scaling_high_freq_factor,
                int32_t rope_scaling_original_max_position_embeddings)
{
    const int64_t rows = q.size(0);
    const int32_t hidden_size = num_attention_heads * head_dim;
    const int32_t kv_hidden_size = num_key_value_heads * head_dim;
    const std::string scaling_type = rope_scaling_type == nullptr ? std::string() : std::string(rope_scaling_type);
    validate_float_tensor_2d(q, rows, hidden_size, "apply_rope::q");
    validate_float_tensor_2d(k, rows, kv_hidden_size, "apply_rope::k");
    validate_int_tensor_1d(positions, rows, "apply_rope::positions");
    if (head_dim <= 0 || head_dim % 2 != 0)
    {
        throw std::runtime_error("apply_rope: head_dim must be a positive even number.");
    }
    if (rope_theta <= 0.0f || rope_scaling_factor <= 0.0f)
    {
        throw std::runtime_error("apply_rope: rope theta and scaling factor must be positive.");
    }
    if (scaling_type == "llama3"
        && (rope_scaling_low_freq_factor <= 0.0f
            || rope_scaling_high_freq_factor <= rope_scaling_low_freq_factor
            || rope_scaling_original_max_position_embeddings <= 0))
    {
        throw std::runtime_error("apply_rope: invalid llama3 rope scaling configuration.");
    }

    if (any_cuda({std::cref(positions), std::cref(q), std::cref(k)}))
    {
        run_apply_rope_device(
            positions,
            q,
            k,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rope_theta,
            scaling_type,
            rope_scaling_factor,
            rope_scaling_low_freq_factor,
            rope_scaling_high_freq_factor,
            rope_scaling_original_max_position_embeddings);
        return;
    }

    validate_cpu_tensor(positions, "apply_rope");
    validate_cpu_tensor(q, "apply_rope");
    validate_cpu_tensor(k, "apply_rope");

    const int32_t* positions_ptr = static_cast<const int32_t*>(tensor_data(positions));
    float* q_ptr = static_cast<float*>(tensor_data(q));
    float* k_ptr = static_cast<float*>(tensor_data(k));

    for (int64_t row = 0; row < rows; ++row)
    {
        const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(hidden_size);
        const float position = static_cast<float>(positions_ptr[row]);
        for (int32_t head = 0; head < num_attention_heads; ++head)
        {
            const int32_t head_offset = head * head_dim;
            const int32_t rotary_half = head_dim / 2;
            for (int32_t dim = 0; dim < rotary_half; ++dim)
            {
                const int32_t idx0 = head_offset + dim;
                const int32_t idx1 = head_offset + rotary_half + dim;
                float inv_freq = 1.0f / std::pow(
                    rope_theta,
                    static_cast<float>(2 * dim) / static_cast<float>(head_dim));
                if (scaling_type == "llama3")
                {
                    const float wavelen = (2.0f * static_cast<float>(M_PI)) / inv_freq;
                    const float low_freq_wavelen =
                        static_cast<float>(rope_scaling_original_max_position_embeddings)
                        / rope_scaling_low_freq_factor;
                    const float high_freq_wavelen =
                        static_cast<float>(rope_scaling_original_max_position_embeddings)
                        / rope_scaling_high_freq_factor;
                    if (wavelen > low_freq_wavelen)
                    {
                        inv_freq /= rope_scaling_factor;
                    }
                    else if (wavelen >= high_freq_wavelen)
                    {
                        const float smooth_factor =
                            (static_cast<float>(rope_scaling_original_max_position_embeddings) / wavelen
                             - rope_scaling_low_freq_factor)
                            / (rope_scaling_high_freq_factor - rope_scaling_low_freq_factor);
                        inv_freq =
                            (1.0f - smooth_factor) * (inv_freq / rope_scaling_factor) + smooth_factor * inv_freq;
                    }
                }
                else if (!scaling_type.empty())
                {
                    inv_freq /= rope_scaling_factor;
                }
                const float theta = position * inv_freq;
                const float cos_theta = std::cos(theta);
                const float sin_theta = std::sin(theta);

                const size_t q0 = row_offset + static_cast<size_t>(idx0);
                const size_t q1 = row_offset + static_cast<size_t>(idx1);
                const float qv0 = q_ptr[q0];
                const float qv1 = q_ptr[q1];
                q_ptr[q0] = qv0 * cos_theta - qv1 * sin_theta;
                q_ptr[q1] = qv1 * cos_theta + qv0 * sin_theta;
            }
        }

        const size_t kv_row_offset = static_cast<size_t>(row) * static_cast<size_t>(kv_hidden_size);
        for (int32_t head = 0; head < num_key_value_heads; ++head)
        {
            const int32_t head_offset = head * head_dim;
            const int32_t rotary_half = head_dim / 2;
            for (int32_t dim = 0; dim < rotary_half; ++dim)
            {
                const int32_t idx0 = head_offset + dim;
                const int32_t idx1 = head_offset + rotary_half + dim;
                float inv_freq = 1.0f / std::pow(
                    rope_theta,
                    static_cast<float>(2 * dim) / static_cast<float>(head_dim));
                if (scaling_type == "llama3")
                {
                    const float wavelen = (2.0f * static_cast<float>(M_PI)) / inv_freq;
                    const float low_freq_wavelen =
                        static_cast<float>(rope_scaling_original_max_position_embeddings)
                        / rope_scaling_low_freq_factor;
                    const float high_freq_wavelen =
                        static_cast<float>(rope_scaling_original_max_position_embeddings)
                        / rope_scaling_high_freq_factor;
                    if (wavelen > low_freq_wavelen)
                    {
                        inv_freq /= rope_scaling_factor;
                    }
                    else if (wavelen >= high_freq_wavelen)
                    {
                        const float smooth_factor =
                            (static_cast<float>(rope_scaling_original_max_position_embeddings) / wavelen
                             - rope_scaling_low_freq_factor)
                            / (rope_scaling_high_freq_factor - rope_scaling_low_freq_factor);
                        inv_freq =
                            (1.0f - smooth_factor) * (inv_freq / rope_scaling_factor) + smooth_factor * inv_freq;
                    }
                }
                else if (!scaling_type.empty())
                {
                    inv_freq /= rope_scaling_factor;
                }
                const float theta = position * inv_freq;
                const float cos_theta = std::cos(theta);
                const float sin_theta = std::sin(theta);

                const size_t k0 = kv_row_offset + static_cast<size_t>(idx0);
                const size_t k1 = kv_row_offset + static_cast<size_t>(idx1);
                const float kv0 = k_ptr[k0];
                const float kv1 = k_ptr[k1];
                k_ptr[k0] = kv0 * cos_theta - kv1 * sin_theta;
                k_ptr[k1] = kv1 * cos_theta + kv0 * sin_theta;
            }
        }
    }
}

void apply_rope(const Tensor& positions,
                Tensor& q,
                Tensor& k,
                int32_t num_attention_heads,
                int32_t num_key_value_heads,
                int32_t head_dim,
                const Tensor& inv_freq)
{
    const int64_t rows = q.size(0);
    const int32_t hidden_size = num_attention_heads * head_dim;
    const int32_t kv_hidden_size = num_key_value_heads * head_dim;
    const int32_t rotary_half = head_dim / 2;
    validate_float_tensor_2d(q, rows, hidden_size, "apply_rope::q");
    validate_float_tensor_2d(k, rows, kv_hidden_size, "apply_rope::k");
    validate_int_tensor_1d(positions, rows, "apply_rope::positions");
    if (head_dim <= 0 || head_dim % 2 != 0)
    {
        throw std::runtime_error("apply_rope: head_dim must be a positive even number.");
    }
    validate_float_tensor_1d(inv_freq, rotary_half, "apply_rope::inv_freq");

    if (any_cuda({std::cref(positions), std::cref(q), std::cref(k), std::cref(inv_freq)}))
    {
        run_apply_rope_device_with_inv_freq(
            positions,
            q,
            k,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            inv_freq);
        return;
    }

    validate_cpu_tensor(positions, "apply_rope");
    validate_cpu_tensor(q, "apply_rope");
    validate_cpu_tensor(k, "apply_rope");
    validate_cpu_tensor(inv_freq, "apply_rope");

    const Tensor inv_freq_contiguous = inv_freq.contiguous();
    const int32_t* positions_ptr = static_cast<const int32_t*>(tensor_data(positions));
    const float* inv_freq_ptr = static_cast<const float*>(tensor_data(inv_freq_contiguous));
    float* q_ptr = static_cast<float*>(tensor_data(q));
    float* k_ptr = static_cast<float*>(tensor_data(k));

    for (int64_t row = 0; row < rows; ++row)
    {
        const float position = static_cast<float>(positions_ptr[row]);
        const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(hidden_size);
        for (int32_t head = 0; head < num_attention_heads; ++head)
        {
            const int32_t head_offset = head * head_dim;
            for (int32_t dim = 0; dim < rotary_half; ++dim)
            {
                const float theta = position * inv_freq_ptr[dim];
                const float cos_theta = std::cos(theta);
                const float sin_theta = std::sin(theta);
                const int32_t idx0 = head_offset + dim;
                const int32_t idx1 = head_offset + rotary_half + dim;
                const size_t q0 = row_offset + static_cast<size_t>(idx0);
                const size_t q1 = row_offset + static_cast<size_t>(idx1);
                const float qv0 = q_ptr[q0];
                const float qv1 = q_ptr[q1];
                q_ptr[q0] = qv0 * cos_theta - qv1 * sin_theta;
                q_ptr[q1] = qv1 * cos_theta + qv0 * sin_theta;
            }
        }

        const size_t kv_row_offset = static_cast<size_t>(row) * static_cast<size_t>(kv_hidden_size);
        for (int32_t head = 0; head < num_key_value_heads; ++head)
        {
            const int32_t head_offset = head * head_dim;
            for (int32_t dim = 0; dim < rotary_half; ++dim)
            {
                const float theta = position * inv_freq_ptr[dim];
                const float cos_theta = std::cos(theta);
                const float sin_theta = std::sin(theta);
                const int32_t idx0 = head_offset + dim;
                const int32_t idx1 = head_offset + rotary_half + dim;
                const size_t k0 = kv_row_offset + static_cast<size_t>(idx0);
                const size_t k1 = kv_row_offset + static_cast<size_t>(idx1);
                const float kv0 = k_ptr[k0];
                const float kv1 = k_ptr[k1];
                k_ptr[k0] = kv0 * cos_theta - kv1 * sin_theta;
                k_ptr[k1] = kv1 * cos_theta + kv0 * sin_theta;
            }
        }
    }
}

void apply_rope(const Tensor& positions,
                Tensor& q,
                Tensor& k,
                int32_t num_attention_heads,
                int32_t num_key_value_heads,
                int32_t head_dim,
                const Tensor& cos_cache,
                const Tensor& sin_cache)
{
    const int64_t rows = q.size(0);
    const int32_t hidden_size = num_attention_heads * head_dim;
    const int32_t kv_hidden_size = num_key_value_heads * head_dim;
    const int32_t rotary_half = head_dim / 2;
    validate_float_tensor_2d(q, rows, hidden_size, "apply_rope::q");
    validate_float_tensor_2d(k, rows, kv_hidden_size, "apply_rope::k");
    validate_int_tensor_1d(positions, rows, "apply_rope::positions");
    if (head_dim <= 0 || head_dim % 2 != 0)
    {
        throw std::runtime_error("apply_rope: head_dim must be a positive even number.");
    }
    if (!cos_cache.defined() || !sin_cache.defined()
        || tensor_dtype(cos_cache) != DType::kFloat32
        || tensor_dtype(sin_cache) != DType::kFloat32
        || cos_cache.dim() != 2
        || sin_cache.dim() != 2
        || cos_cache.size(1) != rotary_half
        || sin_cache.size(1) != rotary_half
        || cos_cache.size(0) != sin_cache.size(0))
    {
        throw std::runtime_error("apply_rope: cos/sin cache shape mismatch.");
    }

    if (any_cuda({std::cref(positions), std::cref(q), std::cref(k), std::cref(cos_cache), std::cref(sin_cache)}))
    {
        run_apply_rope_device_with_cache(
            positions,
            q,
            k,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            cos_cache,
            sin_cache);
        return;
    }

    validate_cpu_tensor(positions, "apply_rope");
    validate_cpu_tensor(q, "apply_rope");
    validate_cpu_tensor(k, "apply_rope");
    validate_cpu_tensor(cos_cache, "apply_rope");
    validate_cpu_tensor(sin_cache, "apply_rope");

    const Tensor cos_contiguous = cos_cache.contiguous();
    const Tensor sin_contiguous = sin_cache.contiguous();
    const int32_t* positions_ptr = static_cast<const int32_t*>(tensor_data(positions));
    const float* cos_ptr = static_cast<const float*>(tensor_data(cos_contiguous));
    const float* sin_ptr = static_cast<const float*>(tensor_data(sin_contiguous));
    float* q_ptr = static_cast<float*>(tensor_data(q));
    float* k_ptr = static_cast<float*>(tensor_data(k));

    for (int64_t row = 0; row < rows; ++row)
    {
        const int32_t position = positions_ptr[row];
        if (position < 0 || position >= cos_cache.size(0))
        {
            throw std::runtime_error("apply_rope: position exceeds cos/sin cache length.");
        }
        const float* cos_row = cos_ptr + static_cast<size_t>(position) * static_cast<size_t>(rotary_half);
        const float* sin_row = sin_ptr + static_cast<size_t>(position) * static_cast<size_t>(rotary_half);
        const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(hidden_size);
        for (int32_t head = 0; head < num_attention_heads; ++head)
        {
            const int32_t head_offset = head * head_dim;
            for (int32_t dim = 0; dim < rotary_half; ++dim)
            {
                const int32_t idx0 = head_offset + dim;
                const int32_t idx1 = head_offset + rotary_half + dim;
                const size_t q0 = row_offset + static_cast<size_t>(idx0);
                const size_t q1 = row_offset + static_cast<size_t>(idx1);
                const float qv0 = q_ptr[q0];
                const float qv1 = q_ptr[q1];
                q_ptr[q0] = qv0 * cos_row[dim] - qv1 * sin_row[dim];
                q_ptr[q1] = qv1 * cos_row[dim] + qv0 * sin_row[dim];
            }
        }

        const size_t kv_row_offset = static_cast<size_t>(row) * static_cast<size_t>(kv_hidden_size);
        for (int32_t head = 0; head < num_key_value_heads; ++head)
        {
            const int32_t head_offset = head * head_dim;
            for (int32_t dim = 0; dim < rotary_half; ++dim)
            {
                const int32_t idx0 = head_offset + dim;
                const int32_t idx1 = head_offset + rotary_half + dim;
                const size_t k0 = kv_row_offset + static_cast<size_t>(idx0);
                const size_t k1 = kv_row_offset + static_cast<size_t>(idx1);
                const float kv0 = k_ptr[k0];
                const float kv1 = k_ptr[k1];
                k_ptr[k0] = kv0 * cos_row[dim] - kv1 * sin_row[dim];
                k_ptr[k1] = kv1 * cos_row[dim] + kv0 * sin_row[dim];
            }
        }
    }
}

void silu_multiply(const Tensor& gate, const Tensor& up, Tensor& out)
{
    if (tensor_dtype(gate) != DType::kFloat32 || tensor_dtype(up) != DType::kFloat32
        || tensor_dtype(out) != DType::kFloat32)
    {
        throw std::runtime_error("silu_multiply: tensors must be float32.");
    }
    if (tensor_shape(gate) != tensor_shape(up) || tensor_shape(gate) != tensor_shape(out))
    {
        throw std::runtime_error("silu_multiply: tensor shapes must match.");
    }
    if (any_cuda({std::cref(gate), std::cref(up), std::cref(out)}))
    {
        validate_same_device({std::cref(gate), std::cref(up), std::cref(out)}, "silu_multiply");
#if TINYLLM_ENABLE_CUDA
        if (gate.device().is_cuda()
            && gate.is_contiguous()
            && up.is_contiguous()
            && out.is_contiguous())
        {
            cuda::launch_silu_multiply_f32(
                gate.data_ptr<float>(),
                up.data_ptr<float>(),
                out.data_ptr<float>(),
                static_cast<int64_t>(tensor_numel(gate)),
                at::cuda::getCurrentCUDAStream(gate.device().index()));
            return;
        }
#endif
        out.copy_((gate / (1.0f + torch::exp(-gate))) * up);
        return;
    }

    validate_cpu_tensor(gate, "silu_multiply");
    validate_cpu_tensor(up, "silu_multiply");
    validate_cpu_tensor(out, "silu_multiply");

    const size_t count = tensor_numel(gate);
    const float* gate_ptr = static_cast<const float*>(tensor_data(gate));
    const float* up_ptr = static_cast<const float*>(tensor_data(up));
    float* out_ptr = static_cast<float*>(tensor_data(out));
    for (size_t i = 0; i < count; ++i)
    {
        const float gate_value = gate_ptr[i];
        const float silu = gate_value / (1.0f + std::exp(-gate_value));
        out_ptr[i] = silu * up_ptr[i];
    }
}

void copy_tensor(const Tensor& src, Tensor& dst)
{
    if (tensor_dtype(src) != DType::kFloat32 || tensor_dtype(dst) != DType::kFloat32)
    {
        throw std::runtime_error("copy_tensor: tensors must be float32.");
    }
    if (tensor_shape(src) != tensor_shape(dst))
    {
        throw std::runtime_error("copy_tensor: tensor shapes must match.");
    }
    if (any_cuda({std::cref(src), std::cref(dst)}))
    {
        validate_same_device({std::cref(src), std::cref(dst)}, "copy_tensor");
        dst.copy_(src);
        return;
    }

    validate_cpu_tensor(src, "copy_tensor");
    validate_cpu_tensor(dst, "copy_tensor");

    const size_t count = tensor_numel(src);
    const float* src_ptr = static_cast<const float*>(tensor_data(src));
    float* dst_ptr = static_cast<float*>(tensor_data(dst));
    for (size_t i = 0; i < count; ++i)
    {
        dst_ptr[i] = src_ptr[i];
    }
}

void add_tensors(const Tensor& lhs, const Tensor& rhs, Tensor& out)
{
    if (tensor_dtype(lhs) != DType::kFloat32 || tensor_dtype(rhs) != DType::kFloat32
        || tensor_dtype(out) != DType::kFloat32)
    {
        throw std::runtime_error("add_tensors: tensors must be float32.");
    }
    if (tensor_shape(lhs) != tensor_shape(rhs) || tensor_shape(lhs) != tensor_shape(out))
    {
        throw std::runtime_error("add_tensors: tensor shapes must match.");
    }
    if (any_cuda({std::cref(lhs), std::cref(rhs), std::cref(out)}))
    {
        validate_same_device({std::cref(lhs), std::cref(rhs), std::cref(out)}, "add_tensors");
#if TINYLLM_ENABLE_CUDA
        if (lhs.device().is_cuda()
            && lhs.is_contiguous()
            && rhs.is_contiguous()
            && out.is_contiguous())
        {
            cuda::launch_add_f32(
                lhs.data_ptr<float>(),
                rhs.data_ptr<float>(),
                out.data_ptr<float>(),
                static_cast<int64_t>(tensor_numel(lhs)),
                at::cuda::getCurrentCUDAStream(lhs.device().index()));
            return;
        }
#endif
        out.copy_(lhs + rhs);
        return;
    }

    validate_cpu_tensor(lhs, "add_tensors");
    validate_cpu_tensor(rhs, "add_tensors");
    validate_cpu_tensor(out, "add_tensors");

    const size_t count = tensor_numel(lhs);
    const float* lhs_ptr = static_cast<const float*>(tensor_data(lhs));
    const float* rhs_ptr = static_cast<const float*>(tensor_data(rhs));
    float* out_ptr = static_cast<float*>(tensor_data(out));
    for (size_t i = 0; i < count; ++i)
    {
        out_ptr[i] = lhs_ptr[i] + rhs_ptr[i];
    }
}

} // namespace ops
} // namespace tiny_llm
