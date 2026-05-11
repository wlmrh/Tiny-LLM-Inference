#include "tiny_llm/core/context.h"
#include "tiny_llm/models/modules/embedding.h"
#include "tiny_llm/models/modules/linear.h"
#include "tiny_llm/models/modules/rotary_embedding.h"
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
    tiny_llm::ExecutionContext ctx(nullptr, nullptr, nullptr);

    tiny_llm::modules::Embedding embedding(3, 2);
    tiny_llm::Tensor embedding_weight = torch::tensor(
        {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor ids = torch::tensor({2, 0}, torch::TensorOptions().dtype(torch::kInt32));
    tiny_llm::Tensor embedded = torch::empty({2, 2}, torch::TensorOptions().dtype(torch::kFloat32));
    embedding.bind_weight(embedding_weight);
    embedding.forward(ids, embedded);
    const float* embedded_ptr = embedded.data_ptr<float>();
    expect_near(embedded_ptr[0], 5.0f, "embedding row 0 col 0 mismatch.");
    expect_near(embedded_ptr[3], 2.0f, "embedding row 1 col 1 mismatch.");

    tiny_llm::Tensor gate = torch::tensor({{0.0f, 1.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor up = torch::tensor({{2.0f, 3.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor activated = torch::empty_like(gate);
    tiny_llm::ops::silu_multiply(gate, up, activated);
    const float* activated_ptr = activated.data_ptr<float>();
    expect_near(activated_ptr[0], 0.0f, "silu_and_mul col 0 mismatch.");
    expect_near(activated_ptr[1], (1.0f / (1.0f + std::exp(-1.0f))) * 3.0f, "silu_and_mul col 1 mismatch.");

    tiny_llm::modules::RotaryEmbedding rotary(1, 1, 2, 10000.0f);
    tiny_llm::Tensor positions = torch::tensor({0}, torch::TensorOptions().dtype(torch::kInt32));
    tiny_llm::Tensor q = torch::tensor({{7.0f, 8.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor k = torch::tensor({{9.0f, 10.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    rotary.forward(positions, q, k);
    expect_near(q.data_ptr<float>()[0], 7.0f, "rotary q position 0 mismatch.");
    expect_near(k.data_ptr<float>()[1], 10.0f, "rotary k position 0 mismatch.");

    tiny_llm::modules::Linear lm_head(2, 2);
    tiny_llm::Tensor lm_weight = torch::tensor(
        {{1.0f, 2.0f}, {3.0f, 4.0f}},
        torch::TensorOptions().dtype(torch::kFloat32));
    lm_head.bind_weight(lm_weight, tiny_llm::modules::WeightLayout::kOutIn);
    tiny_llm::Tensor hidden = torch::tensor({{1.0f, 1.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
    tiny_llm::Tensor logits = lm_head.forward(hidden, ctx);
    expect_near(logits.data_ptr<float>()[0], 3.0f, "lm_head col 0 mismatch.");
    expect_near(logits.data_ptr<float>()[1], 7.0f, "lm_head col 1 mismatch.");

    return 0;
}
