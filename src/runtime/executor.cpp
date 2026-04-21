#include "tiny_llm/runtime/executor.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/models/mini_llama.h"
#include "tiny_llm/models/model.h"
#include "tiny_llm/models/tiny_lm.h"
#include "tiny_llm/operators/paged_attention.h"
#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/execution_context.h"
#include "tiny_llm/runtime/kv_cache.h"
#include "tiny_llm/runtime/sampling.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tiny_llm {

ModelExecutor::ModelExecutor(Model* model, ExecutionContext* ctx, KVCache* kv)
    : model_(model), kv_(kv)
{
    if (kv_ != nullptr)
    {
        kv_block_size_tokens_ = kv_->block_size_tokens();
    }
    set_global_execution_context(ctx);
}

ModelExecutor::ModelExecutor(const EngineArgs& args, KVCache* kv)
    : kv_(kv)
{
    init_from_args(args);
}

ModelExecutor::~ModelExecutor()
{
    reset_global_execution_context();
}

void ModelExecutor::init_from_args(const EngineArgs& args)
{
    if (args.kv_block_size_tokens <= 0)
    {
        throw std::runtime_error("ModelExecutor: kv_block_size_tokens must be positive.");
    }
    kv_block_size_tokens_ = args.kv_block_size_tokens;

    model_ = args.model;
    if (model_ == nullptr)
    {
        switch (args.model_type)
        {
            case EngineModelType::kTinyEmbeddingLM:
                if (args.tiny_lm_checkpoint_path.empty())
                {
                    throw std::runtime_error("ModelExecutor: tiny_lm_checkpoint_path must be provided when model_type is kTinyEmbeddingLM.");
                }
                owned_model_ = std::make_unique<TinyEmbeddingLM>(
                    TinyEmbeddingLM::from_checkpoint(args.tiny_lm_checkpoint_path));
                break;
            case EngineModelType::kMiniLLaMA:
                owned_model_ = std::make_unique<MiniLLaMA>(args.mini_llama_config);
                break;
            case EngineModelType::kPrebuilt:
            default:
                throw std::runtime_error("ModelExecutor: model pointer is null and no constructible model_type is configured.");
        }

        model_ = owned_model_.get();
    }

    initialize_global_execution_context(args, kv_);
}

void ModelExecutor::validate_handles() const
{
    if (model_ == nullptr || g_execution_context == nullptr)
    {
        throw std::runtime_error("ModelExecutor: model/context must be non-null.");
    }
}

int32_t ModelExecutor::vocab_size() const
{
    validate_handles();
    return model_->vocab_size();
}

ModelRunnerOutput ModelExecutor::execute_model(const SchedulerOutput& scheduler_output)
{
    validate_handles();

    struct PreparedInputs {
        torch::Tensor input_tokens; // [num_total_tokens]
        torch::Tensor position_ids; // [num_total_tokens]
        torch::Tensor slot_mapping; // [num_total_tokens]
        torch::Tensor context_lens; // [num_seqs]
        torch::Tensor block_tables; // [num_seqs, max_blocks_per_seq]
    };

    std::vector<uint64_t> req_ids;
    std::vector<int32_t> req_end_offsets;
    std::vector<int32_t> core_seq_ids;

    auto prepare_input_tensors = [&](const SchedulerOutput& output) -> PreparedInputs {
        PreparedInputs prepared;
        const auto int_options = torch::TensorOptions()
            .dtype(to_torch_scalar_type(DType::kInt32))
            .device(c10::kCPU);

        // 调度的总 Sequence 数量
        const int64_t request_count = static_cast<int64_t>(output.scheduled_reqs.size());
        // 需要计算的总 token 数量
        const int64_t total_tokens = static_cast<int64_t>(std::max(0, output.total_num_scheduled_tokens));

        if (request_count == 0 || total_tokens == 0)
        {
            prepared.input_tokens = torch::empty({0}, int_options);
            prepared.position_ids = torch::empty({0}, int_options);
            prepared.slot_mapping = torch::empty({0}, int_options);
            prepared.context_lens = torch::empty({0}, int_options);
            prepared.block_tables = torch::empty({0, 0}, int_options);
            return prepared;
        }

        const int32_t block_size_tokens = kv_block_size_tokens_; // 每个 block 中的 token 数量
        if (block_size_tokens <= 0)
        {
            throw std::runtime_error("ModelExecutor::execute_model: kv block_size_tokens must be positive.");
        }

        int64_t max_blocks_per_seq = 0; // 所有 Request 中，分配到的最多的 block 数量
        int64_t checked_total_tokens = 0; // 已经检查过合法的 token 数量

        req_ids.reserve(static_cast<size_t>(request_count));
        req_end_offsets.reserve(static_cast<size_t>(request_count));
        core_seq_ids.reserve(static_cast<size_t>(total_tokens));

        for (const RequestData& req_data : output.scheduled_reqs)
        {
            const auto count_it = output.num_scheduled_tokens.find(req_data.req_id);
            if (count_it == output.num_scheduled_tokens.end())
            {
                throw std::runtime_error("ModelExecutor::execute_model: missing token budget for scheduled request.");
            }

            const int32_t scheduled_tokens = count_it->second; // req_data 请求需要计算的 token 数量
            if (scheduled_tokens <= 0)
            {
                throw std::runtime_error("ModelExecutor::execute_model: scheduled token budget must be positive.");
            }

            if (req_data.req_id > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
            {
                throw std::runtime_error("ModelExecutor::execute_model: request id exceeds core_seq_id range.");
            }

            if (req_data.num_computed_tokens < 0)
            {
                throw std::runtime_error("ModelExecutor::execute_model: num_computed_tokens must be non-negative.");
            }

            if (req_data.new_token_ids.size() < static_cast<size_t>(scheduled_tokens))
            {
                throw std::runtime_error("ModelExecutor::execute_model: new_token_ids is shorter than scheduled token budget.");
            }

            if (req_data.block_ids.empty())
            {
                throw std::runtime_error("ModelExecutor::execute_model: block_ids must be non-empty for paged-attention metadata.");
            }

            checked_total_tokens += scheduled_tokens;
            max_blocks_per_seq = std::max(max_blocks_per_seq, static_cast<int64_t>(req_data.block_ids.size()));
        }

        if (checked_total_tokens != total_tokens)
        {
            throw std::runtime_error("ModelExecutor::execute_model: total_num_scheduled_tokens mismatches per-request token budget sum.");
        }

        prepared.input_tokens = torch::empty({total_tokens}, int_options);
        prepared.position_ids = torch::empty({total_tokens}, int_options);
        prepared.slot_mapping = torch::empty({total_tokens}, int_options);
        prepared.context_lens = torch::empty({request_count}, int_options);
        prepared.block_tables = torch::full({request_count, max_blocks_per_seq}, -1, int_options);

        int32_t* input_ptr = prepared.input_tokens.data_ptr<int32_t>(); ///< input_ptr[i]: 展平后的 token ids
        int32_t* pos_ptr = prepared.position_ids.data_ptr<int32_t>(); ///< pos_ptr[i]: input_ptr 中第 i 个 token 在其请求中的位置
        int32_t* slot_ptr = prepared.slot_mapping.data_ptr<int32_t>(); ///< slot_ptr[i]: input_ptr 中第 i 个 token 在显存池中的位置
        int32_t* context_ptr = prepared.context_lens.data_ptr<int32_t>(); ///< context_ptr[i]: 第 i 个请求完整长度
        int32_t* block_table_ptr = prepared.block_tables.data_ptr<int32_t>(); ///< 

        int64_t flat_token_index = 0;
        int64_t seq_index = 0; // 计数器，表示当前 ResquestData 是 SchedulerOutput 中的第几个请求

        for (const RequestData& req_data : output.scheduled_reqs)
        {
            const int32_t scheduled_tokens = output.num_scheduled_tokens.at(req_data.req_id);
            const int32_t core_seq_id = static_cast<int32_t>(req_data.req_id);

            req_ids.push_back(req_data.req_id);

            context_ptr[seq_index] = req_data.num_computed_tokens + scheduled_tokens;
            // 遍历当前 req_data 拥有的每一个物理 block，为其分配一个逻辑块号并映射到其真正的物理块号
            for (size_t col = 0; col < req_data.block_ids.size(); ++col)
            {
                // 将当前 Sequence 在 scheduler_output 中的全局 seq 编号映射到物理块编号
                block_table_ptr[seq_index * max_blocks_per_seq + static_cast<int64_t>(col)] = req_data.block_ids[col];
            }

            // 遍历该 SchedulerOuput 中每个等待计算的 token
            for (int32_t i = 0; i < scheduled_tokens; ++i)
            {
                const int32_t position = req_data.num_computed_tokens + i;
                const int32_t logical_block_index = position / block_size_tokens;
                if (logical_block_index < 0
                    || logical_block_index >= static_cast<int32_t>(req_data.block_ids.size()))
                {
                    throw std::runtime_error("ModelExecutor::execute_model: logical block index is out of range.");
                }

                const int32_t physical_block_id = req_data.block_ids[static_cast<size_t>(logical_block_index)];
                if (physical_block_id < 0)
                {
                    throw std::runtime_error("ModelExecutor::execute_model: physical block id must be non-negative.");
                }

                input_ptr[flat_token_index] = req_data.new_token_ids[static_cast<size_t>(i)];
                pos_ptr[flat_token_index] = position;
                slot_ptr[flat_token_index] = physical_block_id * block_size_tokens + (position % block_size_tokens);
                core_seq_ids.push_back(core_seq_id);
                ++flat_token_index;
            }

            req_end_offsets.push_back(static_cast<int32_t>(flat_token_index)); // 下一个 RequestData 初始 token 位置在全局中的偏移
            ++seq_index;
        }

        if (flat_token_index != total_tokens)
        {
            throw std::runtime_error("ModelExecutor::execute_model: flattened token count mismatches total_num_scheduled_tokens.");
        }

        return prepared;
    };

    PreparedInputs prepared_inputs = prepare_input_tensors(scheduler_output);

    ModelRunnerOutput model_output;
    const size_t total_tasks = req_ids.size();
    model_output.req_ids.reserve(total_tasks);
    model_output.sampled_token_ids.reserve(total_tasks);
    model_output.req_id_to_index.reserve(total_tasks);

    auto append_result = [&](uint64_t req_id, int32_t sampled_token_id) {
        const int32_t index = static_cast<int32_t>(model_output.req_ids.size());
        model_output.req_ids.push_back(req_id);
        model_output.sampled_token_ids.push_back(sampled_token_id);
        model_output.req_id_to_index[req_id] = index;
    };

    if (!prepared_inputs.input_tokens.defined() || prepared_inputs.input_tokens.numel() == 0)
    {
        return model_output;
    }

    std::vector<int32_t> sampled_rows;
    try
    {
        sampled_rows = run_forward_batch(
            prepared_inputs.input_tokens,
            prepared_inputs.position_ids,
            prepared_inputs.slot_mapping,
            prepared_inputs.context_lens,
            prepared_inputs.block_tables,
            core_seq_ids,
            req_end_offsets,
            true);
    }
    catch (const std::exception& ex)
    {
        throw std::runtime_error(std::string("ModelExecutor::execute_model: batched forward failed: ") + ex.what());
    }

    for (size_t i = 0; i < req_ids.size(); ++i)
    {
        const int32_t end = req_end_offsets[i];
        if (end <= 0 || static_cast<size_t>(end) > sampled_rows.size())
        {
            throw std::runtime_error("ModelExecutor::execute_model: invalid request row range.");
        }

        const int32_t sampled_token = sampled_rows[static_cast<size_t>(end - 1)];
        append_result(req_ids[i], sampled_token);
    }

    return model_output;
}

std::vector<int32_t> ModelExecutor::run_forward_batch(const Tensor& input_tokens,
                                                      const Tensor& position_ids,
                                                      const Tensor& slot_mapping,
                                                      const Tensor& context_lens,
                                                      const Tensor& block_tables,
                                                      const std::vector<int32_t>& core_seq_ids,
                                                      const std::vector<int32_t>& req_end_offsets,
                                                      bool need_sampling) const
{
    validate_handles();

    if (!input_tokens.defined() || !position_ids.defined() || !slot_mapping.defined()
        || !context_lens.defined() || !block_tables.defined())
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: all prepared tensors must be defined.");
    }

    if (tensor_dtype(input_tokens) != DType::kInt32
        || tensor_dtype(position_ids) != DType::kInt32
        || tensor_dtype(slot_mapping) != DType::kInt32
        || tensor_dtype(context_lens) != DType::kInt32
        || tensor_dtype(block_tables) != DType::kInt32)
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: prepared tensors must be int32.");
    }

    const std::vector<int64_t> input_shape = tensor_shape(input_tokens);
    const std::vector<int64_t> pos_shape = tensor_shape(position_ids);
    const std::vector<int64_t> slot_shape = tensor_shape(slot_mapping);
    const std::vector<int64_t> context_shape = tensor_shape(context_lens);
    const std::vector<int64_t> block_shape = tensor_shape(block_tables);

    if (input_shape.size() != 1 || pos_shape.size() != 1 || slot_shape.size() != 1)
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: input_tokens/position_ids/slot_mapping must be rank-1.");
    }
    if (context_shape.size() != 1)
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: context_lens must be rank-1.");
    }
    if (block_shape.size() != 2 || block_shape[0] != context_shape[0])
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: block_tables must be rank-2 [num_seqs, max_blocks_per_seq].");
    }

    if (input_shape[0] != pos_shape[0] || input_shape[0] != slot_shape[0])
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: token tensor lengths must match.");
    }

    const int64_t B64 = input_shape[0]; // 输入 id Tensor 的第 0 维大小
    if (B64 != static_cast<int64_t>(core_seq_ids.size()))
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: input size mismatch.");
    }
    if (static_cast<int64_t>(req_end_offsets.size()) != context_shape[0])
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: req_end_offsets size must match number of sequences.");
    }
    if (B64 == 0)
    {
        return {};
    }
    if (B64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: batch size exceeds int32 range.");
    }

    const int32_t B = static_cast<int32_t>(B64);
    const int32_t V = model_->vocab_size();
    ExecutionContext& ctx = require_global_execution_context("ModelExecutor::run_forward_batch");

    const int32_t* input_ptr = input_tokens.data_ptr<int32_t>();
    const int32_t* position_ptr = position_ids.data_ptr<int32_t>();

    // 遍历展平后的每一个 token
    for (int32_t row = 0; row < B; ++row)
    {
        const int32_t token_id = input_ptr[static_cast<size_t>(row)]; // 当前的 token_id
        if (token_id < 0 || token_id >= V)
        {
            throw std::runtime_error("ModelExecutor::run_forward_batch: token is out of model vocab range.");
        }

        const int32_t pos = position_ptr[static_cast<size_t>(row)]; // 当前的 token_id 在其原请求中的位置
        if (pos < 0)
        {
            throw std::runtime_error("ModelExecutor::run_forward_batch: position must be non-negative.");
        }

        const int32_t core_seq_id = core_seq_ids[static_cast<size_t>(row)]; // token 所属 seq 的 id
        if (core_seq_id < 0)
        {
            throw std::runtime_error("ModelExecutor::run_forward_batch: core_seq_id must be non-negative.");
        }
    }

    Tensor logits_tensor = torch::zeros(
        {B, V},
        torch::TensorOptions().dtype(to_torch_scalar_type(DType::kFloat32)).device(c10::kCPU));

    const int32_t block_size_tokens = kv_block_size_tokens_; // 每个 block 中存储的 token 上限
    if (block_size_tokens <= 0)
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: block_size_tokens must be positive.");
    }

    // 将参数写到 context 中
    ops::set_paged_attention_runtime_metadata(slot_mapping, context_lens, block_tables, block_size_tokens);
    struct MetadataGuard {
        ~MetadataGuard()
        {
            ops::clear_paged_attention_runtime_metadata();
        }
    } metadata_guard;
    (void)metadata_guard;

    {
        auto guard = ctx.step_guard();
        (void)guard;
        model_->forward_step(input_tokens, position_ids, logits_tensor, ctx);
    }

    if (!need_sampling)
    {
        return {};
    }

    const float* logits_ptr = logits_tensor.data_ptr<float>();
    std::vector<int32_t> sampled_tokens(static_cast<size_t>(B), -1);
    for (int32_t end_offset : req_end_offsets)
    {
        if (end_offset <= 0 || end_offset > B)
        {
            throw std::runtime_error("ModelExecutor::run_forward_batch: req_end_offsets contains out-of-range value.");
        }

        const int32_t target_row = end_offset - 1;
        const float* target_logits = logits_ptr + static_cast<size_t>(target_row) * static_cast<size_t>(V);
        const int32_t sampled = sample_argmax(target_logits, V);
        if (sampled < 0 || sampled >= V)
        {
            throw std::runtime_error("ModelExecutor::run_forward_batch: sampled token is out of model vocab range.");
        }
        sampled_tokens[static_cast<size_t>(target_row)] = sampled;
    }

    return sampled_tokens;
}

} // namespace tiny_llm
