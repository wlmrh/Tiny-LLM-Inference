#include "tiny_llm/core/context.h"
#include "tiny_llm/models/model.h"
#include "tiny_llm/runtime/model_runner.h"
#include "tiny_llm/runtime/prepared_inputs.h"

#include <gtest/gtest.h>

namespace {
class FakeModel final : public tiny_llm::Model {
public:
    int32_t num_layers() const override { return 2; }
    int32_t vocab_size() const override { return 128; }

    tiny_llm::Tensor forward(const tiny_llm::PreparedInputs& inputs,
                             tiny_llm::RuntimeContext& ctx) override
    {
        tiny_llm::Tensor input_cpu = inputs.input_ids.cpu().contiguous();
        tiny_llm::Tensor logits = torch::full(
            {inputs.input_ids.size(0), vocab_size()},
            -1000.0f,
            torch::TensorOptions().dtype(torch::kFloat32).device(ctx.device()));
        tiny_llm::Tensor logits_cpu = logits.cpu();
        const int32_t* input = input_cpu.data_ptr<int32_t>();
        float* out = logits_cpu.data_ptr<float>();
        for (int64_t row = 0; row < input_cpu.numel(); ++row)
        {
            out[row * vocab_size() + ((input[row] + 1) % vocab_size())] = 1000.0f;
        }
        return logits_cpu.to(ctx.device());
    }
};

class CompactLogitsModel final : public tiny_llm::Model {
public:
    int32_t num_layers() const override { return 2; }
    int32_t vocab_size() const override { return 128; }

    tiny_llm::Tensor forward(const tiny_llm::PreparedInputs& inputs,
                             tiny_llm::RuntimeContext& ctx) override
    {
        tiny_llm::Tensor input_cpu = inputs.input_ids.cpu().contiguous();
        tiny_llm::Tensor logits = torch::full(
            {static_cast<int64_t>(inputs.sample_row_offsets.size()), vocab_size()},
            -1000.0f,
            torch::TensorOptions().dtype(torch::kFloat32).device(ctx.device()));
        tiny_llm::Tensor logits_cpu = logits.cpu();
        const int32_t* input = input_cpu.data_ptr<int32_t>();
        float* out = logits_cpu.data_ptr<float>();
        for (size_t sample_index = 0; sample_index < inputs.sample_row_offsets.size(); ++sample_index)
        {
            const int32_t input_row = inputs.sample_row_offsets[sample_index];
            out[sample_index * vocab_size() + ((input[input_row] + 1) % vocab_size())] = 1000.0f;
        }
        return logits_cpu.to(ctx.device());
    }
};

void expect_tensor_values(const tiny_llm::Tensor& tensor, const std::vector<int32_t>& expected)
{
    tiny_llm::Tensor cpu = tensor.cpu().contiguous();
    ASSERT_EQ(cpu.numel(), static_cast<int64_t>(expected.size()));
    const int32_t* ptr = cpu.data_ptr<int32_t>();
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(ptr[i], expected[i]) << "index " << i;
    }
}

tiny_llm::SchedulerOutput make_valid_output()
{
    tiny_llm::SchedulerOutput output;
    tiny_llm::RequestData req0;
    req0.req_id = 7;
    req0.new_token_ids = {11, 12, 13};
    req0.num_computed_tokens = 5;
    req0.block_tables = {{3}, {4}};

    tiny_llm::RequestData req1;
    req1.req_id = 9;
    req1.new_token_ids = {21};
    req1.num_computed_tokens = 16;
    req1.block_tables = {{5, 6}, {7, 8}};

    output.scheduled_reqs = {req0, req1};
    output.num_scheduled_tokens[7] = 3;
    output.num_scheduled_tokens[9] = 1;
    output.total_num_scheduled_tokens = 4;
    return output;
}
}

TEST(ModelRunnerPreparedInputsTest, FlattensSchedulerOutputIntoRuntimeTensors)
{
    FakeModel model;
    tiny_llm::ExecutionContext exec_ctx(nullptr, nullptr, nullptr);
    tiny_llm::ModelRunner runner(&model, &exec_ctx, nullptr);

    tiny_llm::PreparedInputs prepared = runner.prepare_inputs(make_valid_output());
    EXPECT_EQ(prepared.input_ids.numel(), 4);
    EXPECT_EQ(prepared.context_lens.numel(), 2);
    ASSERT_EQ(prepared.sample_row_offsets.size(), 2u);
    EXPECT_EQ(prepared.sample_row_offsets[0], 2);
    EXPECT_EQ(prepared.sample_row_offsets[1], 3);

    expect_tensor_values(prepared.input_ids, {11, 12, 13, 21});
    expect_tensor_values(prepared.positions, {5, 6, 7, 16});
    expect_tensor_values(prepared.slot_mapping, {3 * 16 + 5, 3 * 16 + 6, 3 * 16 + 7, 6 * 16});
    expect_tensor_values(prepared.seq_indices, {0, 0, 0, 1});
    expect_tensor_values(prepared.context_lens, {8, 17});
    EXPECT_EQ(tiny_llm::tensor_shape(prepared.block_tables), std::vector<int64_t>({2, 2, 2}));
}

TEST(ModelRunnerPreparedInputsTest, SamplesOnlyFinalRowForEachRequest)
{
    FakeModel model;
    tiny_llm::ExecutionContext exec_ctx(nullptr, nullptr, nullptr);
    tiny_llm::ModelRunner runner(&model, &exec_ctx, nullptr);

    tiny_llm::ModelRunnerOutput output = runner.run(make_valid_output());
    EXPECT_EQ(output.req_ids, std::vector<uint64_t>({7, 9}));
    EXPECT_EQ(output.sampled_token_ids, std::vector<int32_t>({14, 22}));
    ASSERT_EQ(output.req_id_to_index.size(), 2u);
    EXPECT_EQ(output.req_id_to_index.at(7), 0);
    EXPECT_EQ(output.req_id_to_index.at(9), 1);
}

TEST(ModelRunnerPreparedInputsTest, AcceptsCompactSampleRowLogits)
{
    CompactLogitsModel model;
    tiny_llm::ExecutionContext exec_ctx(nullptr, nullptr, nullptr);
    tiny_llm::ModelRunner runner(&model, &exec_ctx, nullptr);

    tiny_llm::ModelRunnerOutput output = runner.run(make_valid_output());
    EXPECT_EQ(output.req_ids, std::vector<uint64_t>({7, 9}));
    EXPECT_EQ(output.sampled_token_ids, std::vector<int32_t>({14, 22}));
    ASSERT_EQ(output.req_id_to_index.size(), 2u);
    EXPECT_EQ(output.req_id_to_index.at(7), 0);
    EXPECT_EQ(output.req_id_to_index.at(9), 1);
}

TEST(ModelRunnerPreparedInputsTest, RejectsMalformedSchedulerOutput)
{
    FakeModel model;
    tiny_llm::ExecutionContext exec_ctx(nullptr, nullptr, nullptr);
    tiny_llm::ModelRunner runner(&model, &exec_ctx, nullptr);

    tiny_llm::SchedulerOutput missing_budget = make_valid_output();
    missing_budget.num_scheduled_tokens.erase(7);
    EXPECT_THROW(runner.prepare_inputs(missing_budget), std::runtime_error);

    tiny_llm::SchedulerOutput bad_layers = make_valid_output();
    bad_layers.scheduled_reqs[0].block_tables = {{3}};
    EXPECT_THROW(runner.prepare_inputs(bad_layers), std::runtime_error);

    tiny_llm::SchedulerOutput bad_total = make_valid_output();
    bad_total.total_num_scheduled_tokens = 5;
    EXPECT_THROW(runner.prepare_inputs(bad_total), std::runtime_error);

    tiny_llm::SchedulerOutput bad_block = make_valid_output();
    bad_block.scheduled_reqs[0].block_tables = {{}, {4}};
    EXPECT_THROW(runner.prepare_inputs(bad_block), std::runtime_error);
}

TEST(ModelRunnerPreparedInputsTest, RejectsInvalidTokensDuringRun)
{
    FakeModel model;
    tiny_llm::ExecutionContext exec_ctx(nullptr, nullptr, nullptr);
    tiny_llm::ModelRunner runner(&model, &exec_ctx, nullptr);

    tiny_llm::SchedulerOutput output = make_valid_output();
    output.scheduled_reqs[0].new_token_ids[2] = model.vocab_size();
    EXPECT_THROW(runner.run(output), std::runtime_error);
}
