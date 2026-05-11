#include "tiny_llm/models/model.h"
#include "tiny_llm/runtime/model_runner.h"
#include "tiny_llm/runtime/prepared_inputs.h"

#include <stdexcept>

namespace {

class FakeModel final : public tiny_llm::Model {
public:
    int32_t num_layers() const override { return 2; }
    int32_t vocab_size() const override { return 128; }

    tiny_llm::Tensor forward(const tiny_llm::PreparedInputs& inputs,
                             tiny_llm::RuntimeContext& ctx) override
    {
        return torch::zeros(
            {inputs.input_ids.size(0), vocab_size()},
            torch::TensorOptions().dtype(torch::kFloat32).device(ctx.device()));
    }
};

void expect_eq(int32_t actual, int32_t expected, const char* message)
{
    if (actual != expected)
    {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    FakeModel model;
    tiny_llm::ExecutionContext exec_ctx(nullptr, nullptr, nullptr);
    tiny_llm::ModelRunner runner(&model, &exec_ctx, nullptr);

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

    tiny_llm::PreparedInputs prepared = runner.prepare_inputs(output);
    expect_eq(static_cast<int32_t>(prepared.input_ids.numel()), 4, "input length mismatch.");
    expect_eq(static_cast<int32_t>(prepared.context_lens.numel()), 2, "context length size mismatch.");
    expect_eq(static_cast<int32_t>(prepared.sample_row_offsets.size()), 2, "sample row count mismatch.");
    expect_eq(prepared.sample_row_offsets[0], 2, "first sample row mismatch.");
    expect_eq(prepared.sample_row_offsets[1], 3, "second sample row mismatch.");

    tiny_llm::Tensor input_cpu = prepared.input_ids.cpu().contiguous();
    tiny_llm::Tensor position_cpu = prepared.positions.cpu().contiguous();
    tiny_llm::Tensor slot_cpu = prepared.slot_mapping.cpu().contiguous();
    const int32_t* input = input_cpu.data_ptr<int32_t>();
    const int32_t* position = position_cpu.data_ptr<int32_t>();
    const int32_t* slot = slot_cpu.data_ptr<int32_t>();

    expect_eq(input[0], 11, "input row 0 mismatch.");
    expect_eq(input[3], 21, "input row 3 mismatch.");
    expect_eq(position[0], 5, "position row 0 mismatch.");
    expect_eq(position[2], 7, "position row 2 mismatch.");
    expect_eq(position[3], 16, "position row 3 mismatch.");
    expect_eq(slot[0], 3 * 16 + 5, "slot row 0 mismatch.");
    expect_eq(slot[3], 6 * 16, "slot row 3 mismatch.");

    return 0;
}
