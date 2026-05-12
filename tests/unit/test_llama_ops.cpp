#include "tiny_llm/operators/llama_ops.h"

#include <cmath>
#include <stdexcept>

namespace {

void expect_near(float actual, float expected, const char* message)
{
    if (std::fabs(actual - expected) > 1e-5f)
    {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    tiny_llm::Tensor ids = torch::tensor({2, 0}, torch::TensorOptions().dtype(torch::kInt32));
    tiny_llm::Tensor embedding = torch::tensor(
        {{1.0f, 2.0f},
         {3.0f, 4.0f},
         {5.0f, 6.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor embedded = torch::empty({2, 2}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::ops::embedding_lookup(ids, embedding, embedded, 3, 2, true);
    expect_near(embedded.data_ptr<float>()[0], 5.0f, "embedding row 0 col 0 mismatch.");
    expect_near(embedded.data_ptr<float>()[1], 6.0f, "embedding row 0 col 1 mismatch.");
    expect_near(embedded.data_ptr<float>()[2], 1.0f, "embedding row 1 col 0 mismatch.");
    expect_near(embedded.data_ptr<float>()[3], 2.0f, "embedding row 1 col 1 mismatch.");

    tiny_llm::Tensor transposed_embedding = embedding.transpose(0, 1).contiguous();
    tiny_llm::Tensor embedded_transposed = torch::empty({2, 2}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::ops::embedding_lookup(ids, transposed_embedding, embedded_transposed, 3, 2, false);
    expect_near(embedded_transposed.data_ptr<float>()[0], 5.0f, "transposed embedding row 0 col 0 mismatch.");
    expect_near(embedded_transposed.data_ptr<float>()[3], 2.0f, "transposed embedding row 1 col 1 mismatch.");

    tiny_llm::Tensor qkv = torch::tensor(
        {{1.0f, 2.0f, 3.0f, 4.0f},
         {5.0f, 6.0f, 7.0f, 8.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor q = torch::empty({2, 2}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor k = torch::empty({2, 1}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor v = torch::empty({2, 1}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::ops::split_qkv(qkv, q, k, v, 2, 1);
    expect_near(q.data_ptr<float>()[0], 1.0f, "q split mismatch.");
    expect_near(k.data_ptr<float>()[1], 7.0f, "k split mismatch.");
    expect_near(v.data_ptr<float>()[1], 8.0f, "v split mismatch.");

    tiny_llm::Tensor positions = torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32));
    tiny_llm::Tensor rope_q = torch::tensor({{1.0f, 2.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor rope_k = torch::tensor({{3.0f, 4.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::ops::apply_rope(positions, rope_q, rope_k, 1, 1, 2, 10000.0f);
    const float c = std::cos(1.0f);
    const float s = std::sin(1.0f);
    expect_near(rope_q.data_ptr<float>()[0], 1.0f * c - 2.0f * s, "rope q first mismatch.");
    expect_near(rope_q.data_ptr<float>()[1], 2.0f * c + 1.0f * s, "rope q second mismatch.");
    expect_near(rope_k.data_ptr<float>()[0], 3.0f * c - 4.0f * s, "rope k first mismatch.");
    expect_near(rope_k.data_ptr<float>()[1], 4.0f * c + 3.0f * s, "rope k second mismatch.");

    tiny_llm::Tensor llama3_positions = torch::tensor({8192}, torch::TensorOptions().dtype(torch::kInt32));
    tiny_llm::Tensor llama3_rope_q = torch::tensor({{0.0f, 1.0f, 2.0f, 3.0f}},
                                                   torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor llama3_rope_k = torch::tensor({{4.0f, 5.0f, 6.0f, 7.0f}},
                                                   torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::ops::apply_rope(
        llama3_positions,
        llama3_rope_q,
        llama3_rope_k,
        1,
        1,
        4,
        500000.0f,
        "llama3",
        32.0f,
        1.0f,
        4.0f,
        8192);
    const float base_inv_freq = 1.0f / std::sqrt(500000.0f);
    const float wavelen = (2.0f * static_cast<float>(M_PI)) / base_inv_freq;
    const float smooth_factor = (8192.0f / wavelen - 1.0f) / (4.0f - 1.0f);
    const float inv_freq =
        (1.0f - smooth_factor) * (base_inv_freq / 32.0f) + smooth_factor * base_inv_freq;
    const float scaled_theta = 8192.0f * inv_freq;
    const float scaled_c = std::cos(scaled_theta);
    const float scaled_s = std::sin(scaled_theta);
    expect_near(llama3_rope_q.data_ptr<float>()[1], 1.0f * scaled_c - 3.0f * scaled_s, "llama3 rope q mismatch.");
    expect_near(llama3_rope_q.data_ptr<float>()[3], 3.0f * scaled_c + 1.0f * scaled_s, "llama3 rope q mismatch.");
    expect_near(llama3_rope_k.data_ptr<float>()[1], 5.0f * scaled_c - 7.0f * scaled_s, "llama3 rope k mismatch.");
    expect_near(llama3_rope_k.data_ptr<float>()[3], 7.0f * scaled_c + 5.0f * scaled_s, "llama3 rope k mismatch.");

    tiny_llm::Tensor gate = torch::tensor({{-1.0f, 0.0f, 1.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor up = torch::tensor({{2.0f, 3.0f, 4.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor activated = torch::empty({1, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::ops::silu_multiply(gate, up, activated);
    expect_near(activated.data_ptr<float>()[0], (-1.0f / (1.0f + std::exp(1.0f))) * 2.0f, "silu mismatch.");
    expect_near(activated.data_ptr<float>()[2], (1.0f / (1.0f + std::exp(-1.0f))) * 4.0f, "silu mismatch.");

    tiny_llm::Tensor copied = torch::empty_like(activated);
    tiny_llm::ops::copy_tensor(activated, copied);
    expect_near(copied.data_ptr<float>()[2], activated.data_ptr<float>()[2], "copy mismatch.");

    tiny_llm::Tensor added = torch::empty_like(activated);
    tiny_llm::ops::add_tensors(activated, copied, added);
    expect_near(added.data_ptr<float>()[2], activated.data_ptr<float>()[2] * 2.0f, "add mismatch.");

    return 0;
}
