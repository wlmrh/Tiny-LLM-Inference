#include "tiny_llm/runtime/executor.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/models/hf_safetensors_loader.h"
#include "tiny_llm/models/llama_model.h"
#include "tiny_llm/models/llama_weight_map.h"
#include "tiny_llm/models/mini_llama.h"
#include "tiny_llm/models/model.h"
#include "tiny_llm/models/tiny_lm.h"
#include "tiny_llm/operators/paged_attention.h"
#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/execution_context.h"
#include "tiny_llm/runtime/kv_cache.h"
#include "tiny_llm/runtime/sampling.h"

#if TINYLLM_ENABLE_CUDA
#include "utils/cuda_utils.h"
#endif

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tiny_llm {

namespace {

int64_t checked_numel(const std::vector<int64_t>& shape, const char* caller)
{
    int64_t numel = 1;
    for (int64_t dim : shape)
    {
        if (dim < 0)
        {
            throw std::runtime_error(std::string(caller) + ": tensor shape dimensions must be non-negative.");
        }
        if (dim == 0)
        {
            return 0;
        }
        if (numel > std::numeric_limits<int64_t>::max() / dim)
        {
            throw std::runtime_error(std::string(caller) + ": tensor shape is too large.");
        }
        numel *= dim;
    }
    return numel;
}

Tensor make_int32_tensor_from_host(const std::vector<int32_t>& values,
                                   const std::vector<int64_t>& shape,
                                   const c10::Device& device,
                                   const char* caller)
{
    const int64_t expected_numel = checked_numel(shape, caller);
    if (expected_numel != static_cast<int64_t>(values.size()))
    {
        throw std::runtime_error(std::string(caller) + ": host value count does not match tensor shape.");
    }

    Tensor cpu_tensor = torch::empty(
        shape,
        torch::TensorOptions().dtype(to_torch_scalar_type(DType::kInt32)).device(c10::kCPU));
    if (!values.empty())
    {
        std::memcpy(cpu_tensor.data_ptr<int32_t>(), values.data(), values.size() * sizeof(int32_t));
    }
    if (device.is_cpu())
    {
        return cpu_tensor;
    }
    return cpu_tensor.to(device, /*non_blocking=*/false, /*copy=*/true).contiguous();
}

Tensor tensor_to_cpu_contiguous(const Tensor& tensor)
{
    if (tensor.device().is_cpu())
    {
        return tensor.contiguous();
    }
    return tensor.to(c10::kCPU, /*non_blocking=*/false, /*copy=*/true).contiguous();
}

void synchronize_tensor_if_cuda(const Tensor& tensor, ExecutionContext& ctx, const char* caller)
{
    if (!tensor.defined() || tensor.device().is_cpu())
    {
        return;
    }

#if TINYLLM_ENABLE_CUDA
    (void)caller;
    CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));
#else
    (void)ctx;
    throw std::runtime_error(std::string(caller) + ": CUDA tensor was produced in a CPU-only build.");
#endif
}

} // namespace

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
    owned_hf_loader_.reset();
    args.parallel_config.validate();

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
            case EngineModelType::kHFLlamaSafeTensor:
            {
                if (args.hf_model_dir.empty())
                {
                    throw std::runtime_error(
                        "ModelExecutor: hf_model_dir must be provided when model_type is kHFLlamaSafeTensor.");
                }

                const std::string weight_file = args.hf_weight_file.empty()
                    ? std::string("model.safetensors")
                    : args.hf_weight_file;
                const std::filesystem::path weight_path =
                    std::filesystem::path(args.hf_model_dir) / weight_file;

                const LlamaConfig hf_config = HFLlamaConfigLoader::load_from_dir(args.hf_model_dir);
                owned_hf_loader_ = std::make_unique<HFSafeTensorLoader>(
                    HFSafeTensorLoader::from_file(weight_path.string()));
                WeightMap weight_map = WeightMap::from_safetensors(
                    *owned_hf_loader_,
                    args.parallel_config);
                auto llama_model = std::make_unique<LlamaModel>(hf_config, std::move(weight_map));
                llama_model->allocate_buffers(resolve_model_max_batch_size(args), args.parallel_config);
                owned_model_ = std::move(llama_model);
                break;
            }
            case EngineModelType::kPrebuilt:
            default:
                throw std::runtime_error("ModelExecutor: model pointer is null and no constructible model_type is configured.");
        }

        model_ = owned_model_.get();
    }

    initialize_global_execution_context(args, kv_);
}

int32_t ModelExecutor::resolve_model_max_batch_size(const EngineArgs& args) const
{
    int32_t max_batch_size = args.max_batch_size;
    if (args.scheduler_config.max_prefill_tokens_per_step > max_batch_size)
    {
        max_batch_size = args.scheduler_config.max_prefill_tokens_per_step;
    }
    if (max_batch_size <= 0)
    {
        throw std::runtime_error("ModelExecutor: resolved model max_batch_size must be positive.");
    }
    return max_batch_size;
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
    ExecutionContext& ctx = require_global_execution_context("ModelExecutor::execute_model");
    const c10::Device runtime_device = ctx.device();

    struct PreparedInputs {
        torch::Tensor input_tokens; // [num_total_tokens]
        torch::Tensor position_ids; // [num_total_tokens]
        torch::Tensor slot_mapping; // [num_total_tokens]
        torch::Tensor seq_indices; // [num_total_tokens]
        torch::Tensor context_lens; // [num_seqs]
        torch::Tensor block_tables; // [num_layers, num_seqs, max_blocks_per_seq]
    };

    std::vector<uint64_t> req_ids;
    std::vector<int32_t> req_end_offsets;
    std::vector<int32_t> core_seq_ids;

    auto prepare_input_tensors = [&](const SchedulerOutput& output) -> PreparedInputs {
        PreparedInputs prepared;

        // the number of the sequences that are scheduled
        const int64_t request_count = static_cast<int64_t>(output.scheduled_reqs.size());
        // the number of tokens that needs to be computed
        const int64_t total_tokens = static_cast<int64_t>(std::max(0, output.total_num_scheduled_tokens));

        if (request_count == 0 || total_tokens == 0)
        {
            prepared.input_tokens = make_int32_tensor_from_host({}, {0}, runtime_device, "ModelExecutor::execute_model");
            prepared.position_ids = make_int32_tensor_from_host({}, {0}, runtime_device, "ModelExecutor::execute_model");
            prepared.slot_mapping = make_int32_tensor_from_host({}, {0}, runtime_device, "ModelExecutor::execute_model");
            prepared.seq_indices = make_int32_tensor_from_host({}, {0}, runtime_device, "ModelExecutor::execute_model");
            prepared.context_lens = make_int32_tensor_from_host({}, {0}, runtime_device, "ModelExecutor::execute_model");
            prepared.block_tables = make_int32_tensor_from_host({}, {0, 0, 0}, runtime_device, "ModelExecutor::execute_model");
            return prepared;
        }

        const int32_t block_size_tokens = kv_block_size_tokens_; // the number of tokens stored in each block
        if (block_size_tokens <= 0)
        {
            throw std::runtime_error("ModelExecutor::execute_model: kv block_size_tokens must be positive.");
        }

        int64_t num_layers = 0; // num of layers in the model
        int64_t max_blocks_per_seq = 0; // maximum block the scheduled sequences own 
        int64_t checked_total_tokens = 0; // the amount of token that have been checked for legality

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
            // tokens that current request scheduled
            const int32_t scheduled_tokens = count_it->second;
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

            if (req_data.block_tables.empty())
            {
                throw std::runtime_error("ModelExecutor::execute_model: block_tables must be non-empty for paged-attention metadata.");
            }
            if (num_layers == 0)
            {
                num_layers = static_cast<int64_t>(req_data.block_tables.size());
            }
            else if (num_layers != static_cast<int64_t>(req_data.block_tables.size()))
            {
                throw std::runtime_error("ModelExecutor::execute_model: all requests must have same number of KV layers.");
            }
            for (const std::vector<int32_t>& layer_blocks : req_data.block_tables)
            {
                if (layer_blocks.empty())
                {
                    throw std::runtime_error("ModelExecutor::execute_model: each layer block table must be non-empty.");
                }
                max_blocks_per_seq = std::max(max_blocks_per_seq, static_cast<int64_t>(layer_blocks.size()));
            }

            checked_total_tokens += scheduled_tokens;
        }

        if (checked_total_tokens != total_tokens)
        {
            throw std::runtime_error("ModelExecutor::execute_model: total_num_scheduled_tokens mismatches per-request token budget sum.");
        }

        // Host-side staging buffers for the batched forward pass. The scheduler output is
        // flattened here and then converted into device tensors.
        std::vector<int32_t> input_values(static_cast<size_t>(total_tokens), 0);
        // Logical position of each flattened token within its original sequence.
        std::vector<int32_t> position_values(static_cast<size_t>(total_tokens), 0);
        // Physical KV slot for each flattened token, derived from the request page table.
        std::vector<int32_t> slot_values(static_cast<size_t>(total_tokens), 0);
        // Batch-local sequence index for each flattened token.
        std::vector<int32_t> seq_index_values(static_cast<size_t>(total_tokens), 0);
        // Final context length for each scheduled sequence after appending this step.
        std::vector<int32_t> context_values(static_cast<size_t>(request_count), 0);
        // Dense host-side page table buffer with shape
        // [num_layers, request_count, max_blocks_per_seq]. Unused entries remain -1.
        std::vector<int32_t> block_table_values(
            static_cast<size_t>(num_layers * request_count * max_blocks_per_seq),
            -1);

        int32_t* input_ptr = input_values.data(); // Flattened token ids in execution order.
        int32_t* pos_ptr = position_values.data(); // Logical positions aligned with input_ptr.
        int32_t* slot_ptr = slot_values.data(); // KV slots aligned with input_ptr.
        int32_t* seq_index_ptr = seq_index_values.data(); // Sequence indices aligned with input_ptr.
        int32_t* context_ptr = context_values.data(); // Computed tokens
        int32_t* block_table_ptr = block_table_values.data(); // Flattened [layer, seq, logical_block] page table.

        int64_t flat_token_index = 0; // Write cursor for token-major buffers.
        int64_t seq_index = 0; // Row index of the current request in this batch.

        for (const RequestData& req_data : output.scheduled_reqs)
        {
            const int32_t scheduled_tokens = output.num_scheduled_tokens.at(req_data.req_id);
            const int32_t core_seq_id = static_cast<int32_t>(req_data.req_id);

            req_ids.push_back(req_data.req_id);

            context_ptr[seq_index] = req_data.num_computed_tokens + scheduled_tokens;
            // 遍历当前 req_data 每层拥有的物理 block，为逻辑块号映射真正的物理块号
            for (size_t layer = 0; layer < req_data.block_tables.size(); ++layer)
            {
                const std::vector<int32_t>& layer_blocks = req_data.block_tables[layer];
                for (size_t col = 0; col < layer_blocks.size(); ++col)
                {
                    block_table_ptr[
                        static_cast<int64_t>(layer) * request_count * max_blocks_per_seq
                        + seq_index * max_blocks_per_seq
                        + static_cast<int64_t>(col)] = layer_blocks[col];
                }
            }

            // 遍历该 SchedulerOuput 中每个等待计算的 token
            for (int32_t i = 0; i < scheduled_tokens; ++i)
            {
                const int32_t position = req_data.num_computed_tokens + i;
                const int32_t logical_block_index = position / block_size_tokens;
                if (logical_block_index < 0
                    || logical_block_index >= static_cast<int32_t>(req_data.block_tables[0].size()))
                {
                    throw std::runtime_error("ModelExecutor::execute_model: logical block index is out of range.");
                }

                const int32_t physical_block_id = req_data.block_tables[0][static_cast<size_t>(logical_block_index)];
                if (physical_block_id < 0)
                {
                    throw std::runtime_error("ModelExecutor::execute_model: physical block id must be non-negative.");
                }

                input_ptr[flat_token_index] = req_data.new_token_ids[static_cast<size_t>(i)];
                pos_ptr[flat_token_index] = position;
                slot_ptr[flat_token_index] = physical_block_id * block_size_tokens + (position % block_size_tokens);
                seq_index_ptr[flat_token_index] = static_cast<int32_t>(seq_index);
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

        prepared.input_tokens = make_int32_tensor_from_host(
            input_values,
            {total_tokens},
            runtime_device,
            "ModelExecutor::execute_model");
        prepared.position_ids = make_int32_tensor_from_host(
            position_values,
            {total_tokens},
            runtime_device,
            "ModelExecutor::execute_model");
        prepared.slot_mapping = make_int32_tensor_from_host(
            slot_values,
            {total_tokens},
            runtime_device,
            "ModelExecutor::execute_model");
        prepared.seq_indices = make_int32_tensor_from_host(
            seq_index_values,
            {total_tokens},
            runtime_device,
            "ModelExecutor::execute_model");
        prepared.context_lens = make_int32_tensor_from_host(
            context_values,
            {request_count},
            runtime_device,
            "ModelExecutor::execute_model");
        prepared.block_tables = make_int32_tensor_from_host(
            block_table_values,
            {num_layers, request_count, max_blocks_per_seq},
            runtime_device,
            "ModelExecutor::execute_model");

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
            prepared_inputs.seq_indices,
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
                                                      const Tensor& seq_indices,
                                                      const Tensor& context_lens,
                                                      const Tensor& block_tables,
                                                      const std::vector<int32_t>& core_seq_ids,
                                                      const std::vector<int32_t>& req_end_offsets,
                                                      bool need_sampling) const
{
    validate_handles();

    if (!input_tokens.defined() || !position_ids.defined() || !slot_mapping.defined() || !seq_indices.defined()
        || !context_lens.defined() || !block_tables.defined())
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: all prepared tensors must be defined.");
    }

    if (tensor_dtype(input_tokens) != DType::kInt32
        || tensor_dtype(position_ids) != DType::kInt32
        || tensor_dtype(slot_mapping) != DType::kInt32
        || tensor_dtype(seq_indices) != DType::kInt32
        || tensor_dtype(context_lens) != DType::kInt32
        || tensor_dtype(block_tables) != DType::kInt32)
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: prepared tensors must be int32.");
    }

    const std::vector<int64_t> input_shape = tensor_shape(input_tokens);
    const std::vector<int64_t> pos_shape = tensor_shape(position_ids);
    const std::vector<int64_t> slot_shape = tensor_shape(slot_mapping);
    const std::vector<int64_t> seq_index_shape = tensor_shape(seq_indices);
    const std::vector<int64_t> context_shape = tensor_shape(context_lens);
    const std::vector<int64_t> block_shape = tensor_shape(block_tables);

    if (input_shape.size() != 1 || pos_shape.size() != 1 || slot_shape.size() != 1 || seq_index_shape.size() != 1)
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: input_tokens/position_ids/slot_mapping/seq_indices must be rank-1.");
    }
    if (context_shape.size() != 1)
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: context_lens must be rank-1.");
    }
    if (block_shape.size() != 3 || block_shape[1] != context_shape[0])
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: block_tables must be rank-3 [num_layers, num_seqs, max_blocks_per_seq].");
    }

    if (input_shape[0] != pos_shape[0] || input_shape[0] != slot_shape[0] || input_shape[0] != seq_index_shape[0])
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
    const c10::Device runtime_device = ctx.device();

    const Tensor input_tokens_for_validation = tensor_to_cpu_contiguous(input_tokens);
    const Tensor position_ids_for_validation = tensor_to_cpu_contiguous(position_ids);
    const int32_t* input_ptr = input_tokens_for_validation.data_ptr<int32_t>();
    const int32_t* position_ptr = position_ids_for_validation.data_ptr<int32_t>();

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
        torch::TensorOptions().dtype(to_torch_scalar_type(DType::kFloat32)).device(runtime_device));

    const int32_t block_size_tokens = kv_block_size_tokens_; // 每个 block 中存储的 token 上限
    if (block_size_tokens <= 0)
    {
        throw std::runtime_error("ModelExecutor::run_forward_batch: block_size_tokens must be positive.");
    }

    // 将参数写到 context 中
    ops::set_paged_attention_runtime_metadata(slot_mapping, seq_indices, context_lens, block_tables, block_size_tokens);
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

    synchronize_tensor_if_cuda(logits_tensor, ctx, "ModelExecutor::run_forward_batch");

    std::vector<int32_t> sampled_tokens(static_cast<size_t>(B), -1);
    for (int32_t end_offset : req_end_offsets)
    {
        if (end_offset <= 0 || end_offset > B)
        {
            throw std::runtime_error("ModelExecutor::run_forward_batch: req_end_offsets contains out-of-range value.");
        }

        const int32_t target_row = end_offset - 1;
        Tensor target_logits_tensor = logits_tensor[target_row];
        if (!target_logits_tensor.device().is_cpu())
        {
            target_logits_tensor = tensor_to_cpu_contiguous(target_logits_tensor);
        }
        else
        {
            target_logits_tensor = target_logits_tensor.contiguous();
        }
        const float* target_logits = target_logits_tensor.data_ptr<float>();
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
