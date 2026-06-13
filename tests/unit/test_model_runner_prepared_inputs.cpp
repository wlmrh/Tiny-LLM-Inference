#include "tiny_llm/core/context.h"
#include "tiny_llm/models/model.h"
#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/model_runner.h"
#include "tiny_llm/runtime/prepared_inputs.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {
std::vector<int32_t> copy_int32_tensor(const tiny_llm::Tensor& tensor)
{
    tiny_llm::Tensor cpu = tensor.cpu().contiguous();
    std::vector<int32_t> values(static_cast<size_t>(cpu.numel()));
    const int32_t* ptr = cpu.data_ptr<int32_t>();
    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i] = ptr[i];
    }
    return values;
}

class FakeModel final : public tiny_llm::Model {
public:
    int32_t num_layers() const override { return 2; }
    int32_t vocab_size() const override { return 128; }

    tiny_llm::Tensor forward(const tiny_llm::PreparedInputs& inputs,
                             tiny_llm::RuntimeContext& ctx) override
    {
        capture_inputs(inputs);
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

    int64_t forward_calls = 0;
    std::vector<int32_t> last_input_ids;
    std::vector<int32_t> last_positions;
    std::vector<int32_t> last_slot_mapping;
    std::vector<int32_t> last_seq_indices;
    std::vector<int32_t> last_context_lens;
    std::vector<int64_t> last_block_tables_shape;
    std::vector<int32_t> last_sample_row_offsets;

private:
    void capture_inputs(const tiny_llm::PreparedInputs& inputs)
    {
        ++forward_calls;
        last_input_ids = copy_int32_tensor(inputs.input_ids);
        last_positions = copy_int32_tensor(inputs.positions);
        last_slot_mapping = copy_int32_tensor(inputs.slot_mapping);
        last_seq_indices = copy_int32_tensor(inputs.seq_indices);
        last_context_lens = copy_int32_tensor(inputs.context_lens);
        last_block_tables_shape = tiny_llm::tensor_shape(inputs.block_tables);
        last_sample_row_offsets = inputs.sample_row_offsets;
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

tiny_llm::EngineArgs make_model_runner_args(tiny_llm::Model* model, tiny_llm::ExecutionContext* ctx)
{
    tiny_llm::EngineArgs args;
    args.model = model;
    args.ctx = ctx;
    return args;
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
    tiny_llm::EngineArgs args = make_model_runner_args(&model, &exec_ctx);
    tiny_llm::ModelRunner runner(args, nullptr);

    tiny_llm::ModelRunnerOutput output = runner.run(make_valid_output());
    ASSERT_EQ(output.sampled_token_ids.size(), 2u);
    EXPECT_EQ(model.forward_calls, 1);
    EXPECT_EQ(model.last_sample_row_offsets, std::vector<int32_t>({2, 3}));
    EXPECT_EQ(model.last_input_ids, std::vector<int32_t>({11, 12, 13, 21}));
    EXPECT_EQ(model.last_positions, std::vector<int32_t>({5, 6, 7, 16}));
    EXPECT_EQ(model.last_slot_mapping, std::vector<int32_t>({3 * 16 + 5, 3 * 16 + 6, 3 * 16 + 7, 6 * 16}));
    EXPECT_EQ(model.last_seq_indices, std::vector<int32_t>({0, 0, 0, 1}));
    EXPECT_EQ(model.last_context_lens, std::vector<int32_t>({8, 17}));
    EXPECT_EQ(model.last_block_tables_shape, std::vector<int64_t>({2, 2, 2}));
}

TEST(ModelRunnerPreparedInputsTest, SamplesOnlyFinalRowForEachRequest)
{
    FakeModel model;
    tiny_llm::ExecutionContext exec_ctx(nullptr, nullptr, nullptr);
    tiny_llm::EngineArgs args = make_model_runner_args(&model, &exec_ctx);
    tiny_llm::ModelRunner runner(args, nullptr);

    tiny_llm::ModelRunnerOutput output = runner.run(make_valid_output());
    ASSERT_EQ(output.sampled_token_ids.size(), 2u);
    ASSERT_EQ(output.req_id_to_index.size(), 2u);
    EXPECT_EQ(output.sampled_token_ids[static_cast<size_t>(output.req_id_to_index.at(7))], 14);
    EXPECT_EQ(output.sampled_token_ids[static_cast<size_t>(output.req_id_to_index.at(9))], 22);
}

TEST(ModelRunnerPreparedInputsTest, AcceptsCompactSampleRowLogits)
{
    CompactLogitsModel model;
    tiny_llm::ExecutionContext exec_ctx(nullptr, nullptr, nullptr);
    tiny_llm::EngineArgs args = make_model_runner_args(&model, &exec_ctx);
    tiny_llm::ModelRunner runner(args, nullptr);

    tiny_llm::ModelRunnerOutput output = runner.run(make_valid_output());
    ASSERT_EQ(output.sampled_token_ids.size(), 2u);
    ASSERT_EQ(output.req_id_to_index.size(), 2u);
    EXPECT_EQ(output.sampled_token_ids[static_cast<size_t>(output.req_id_to_index.at(7))], 14);
    EXPECT_EQ(output.sampled_token_ids[static_cast<size_t>(output.req_id_to_index.at(9))], 22);
}

TEST(ModelRunnerPreparedInputsTest, RejectsMalformedSchedulerOutput)
{
    FakeModel model;
    tiny_llm::ExecutionContext exec_ctx(nullptr, nullptr, nullptr);
    tiny_llm::EngineArgs args = make_model_runner_args(&model, &exec_ctx);
    tiny_llm::ModelRunner runner(args, nullptr);

    tiny_llm::SchedulerOutput missing_budget = make_valid_output();
    missing_budget.num_scheduled_tokens.erase(7);
    EXPECT_THROW(runner.run(missing_budget), std::runtime_error);

    tiny_llm::SchedulerOutput bad_layers = make_valid_output();
    bad_layers.scheduled_reqs[0].block_tables = {{3}};
    EXPECT_THROW(runner.run(bad_layers), std::runtime_error);

    tiny_llm::SchedulerOutput bad_total = make_valid_output();
    bad_total.total_num_scheduled_tokens = 5;
    EXPECT_THROW(runner.run(bad_total), std::runtime_error);

    tiny_llm::SchedulerOutput bad_block = make_valid_output();
    bad_block.scheduled_reqs[0].block_tables = {{}, {4}};
    EXPECT_THROW(runner.run(bad_block), std::runtime_error);
}

TEST(ModelRunnerPreparedInputsTest, RejectsInvalidTokensDuringRun)
{
    FakeModel model;
    tiny_llm::ExecutionContext exec_ctx(nullptr, nullptr, nullptr);
    tiny_llm::EngineArgs args = make_model_runner_args(&model, &exec_ctx);
    tiny_llm::ModelRunner runner(args, nullptr);

    tiny_llm::SchedulerOutput output = make_valid_output();
    output.scheduled_reqs[0].new_token_ids[2] = model.vocab_size();
    EXPECT_THROW(runner.run(output), std::runtime_error);
}
