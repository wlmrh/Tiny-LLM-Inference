#include "tiny_llm/runtime/model_runner.h"

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/models/hf_safetensors_loader.h"
#include "tiny_llm/models/llama_model.h"
#include "tiny_llm/models/llama_weight_map.h"
#include "tiny_llm/models/mini_llama.h"
#include "tiny_llm/models/model.h"
#include "tiny_llm/models/tiny_lm.h"
#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/execution_context.h"
#include "tiny_llm/runtime/kv_cache.h"
#include "tiny_llm/runtime/runtime_context.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace tiny_llm {

namespace {

using ProfileClock = std::chrono::steady_clock;

double elapsed_profile_ms(ProfileClock::time_point start, ProfileClock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void synchronize_for_profile(const c10::Device& device)
{
#if TINYLLM_ENABLE_CUDA
    if (device.is_cuda())
    {
        const int device_index = device.index() >= 0 ? static_cast<int>(device.index()) : 0;
        cudaSetDevice(device_index);
        cudaDeviceSynchronize();
    }
#else
    (void)device;
#endif
}

bool runtime_detail_profile_enabled()
{
    const char* value = std::getenv("TINYLLM_PROFILE_DETAIL");
    if (value == nullptr)
    {
        return false;
    }
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

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

std::vector<std::filesystem::path> resolve_hf_safetensor_paths(const EngineArgs& args)
{
    const std::filesystem::path model_dir(args.hf_model_dir);
    std::vector<std::filesystem::path> paths;

    if (!args.hf_weight_file.empty())
    {
        const std::filesystem::path requested = model_dir / args.hf_weight_file;
        if (std::filesystem::exists(requested))
        {
            paths.push_back(requested);
            return paths;
        }
        if (args.hf_weight_file != "model.safetensors")
        {
            throw std::runtime_error("ModelRunner: hf_weight_file does not exist: " + requested.string());
        }
    }

    const std::filesystem::path default_weight = model_dir / "model.safetensors";
    if (std::filesystem::exists(default_weight))
    {
        paths.push_back(default_weight);
        return paths;
    }

    if (!std::filesystem::exists(model_dir) || !std::filesystem::is_directory(model_dir))
    {
        throw std::runtime_error("ModelRunner: hf_model_dir does not exist or is not a directory: " + args.hf_model_dir);
    }

    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(model_dir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        const std::filesystem::path& path = entry.path();
        if (path.extension() == ".safetensors")
        {
            paths.push_back(path);
        }
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty())
    {
        throw std::runtime_error("ModelRunner: no safetensors weight files found in " + args.hf_model_dir);
    }
    return paths;
}

WeightMap load_weight_map_from_safetensors(
    const std::vector<std::filesystem::path>& paths,
    const ParallelConfig& parallel_config,
    std::vector<std::unique_ptr<HFSafeTensorLoader>>& owned_loaders)
{
    WeightMap weight_map;
    parallel_config.validate();
    const c10::Device target_device = parallel_config.torch_device();
    owned_loaders.clear();
    owned_loaders.reserve(paths.size());

    for (const std::filesystem::path& path : paths)
    {
        auto loader = std::make_unique<HFSafeTensorLoader>(HFSafeTensorLoader::from_file(path.string()));
        for (const std::string& key : loader->keys())
        {
            Tensor tensor = loader->tensor(key);
            if (tensor.device() != target_device)
            {
                tensor = tensor.to(target_device, /*non_blocking=*/false, /*copy=*/true).contiguous();
            }
            else if (!tensor.is_contiguous())
            {
                tensor = tensor.contiguous();
            }
            weight_map.add_tensor(key, tensor);
        }
        owned_loaders.push_back(std::move(loader));
    }

    return weight_map;
}

void validate_prepared_tensor_pack(const PreparedInputs& inputs,
                                   int32_t vocab_size,
                                   int32_t num_layers,
                                   const char* caller)
{
    if (!inputs.input_ids.defined()
        || !inputs.positions.defined()
        || !inputs.slot_mapping.defined()
        || !inputs.seq_indices.defined()
        || !inputs.context_lens.defined()
        || !inputs.block_tables.defined())
    {
        throw std::runtime_error(std::string(caller) + ": all prepared tensors must be defined.");
    }
    if (tensor_dtype(inputs.input_ids) != DType::kInt32
        || tensor_dtype(inputs.positions) != DType::kInt32
        || tensor_dtype(inputs.slot_mapping) != DType::kInt32
        || tensor_dtype(inputs.seq_indices) != DType::kInt32
        || tensor_dtype(inputs.context_lens) != DType::kInt32
        || tensor_dtype(inputs.block_tables) != DType::kInt32)
    {
        throw std::runtime_error(std::string(caller) + ": prepared tensors must be int32.");
    }

    const std::vector<int64_t> input_shape = tensor_shape(inputs.input_ids);
    const std::vector<int64_t> pos_shape = tensor_shape(inputs.positions);
    const std::vector<int64_t> slot_shape = tensor_shape(inputs.slot_mapping);
    const std::vector<int64_t> seq_shape = tensor_shape(inputs.seq_indices);
    const std::vector<int64_t> context_shape = tensor_shape(inputs.context_lens);
    const std::vector<int64_t> block_shape = tensor_shape(inputs.block_tables);
    if (input_shape.size() != 1 || pos_shape.size() != 1 || slot_shape.size() != 1 || seq_shape.size() != 1)
    {
        throw std::runtime_error(std::string(caller) + ": token-aligned tensors must be rank-1.");
    }
    if (input_shape[0] != pos_shape[0] || input_shape[0] != slot_shape[0] || input_shape[0] != seq_shape[0])
    {
        throw std::runtime_error(std::string(caller) + ": token-aligned tensor lengths must match.");
    }
    if (context_shape.size() != 1)
    {
        throw std::runtime_error(std::string(caller) + ": context_lens must be rank-1.");
    }
    if (block_shape.size() != 3 || block_shape[0] != num_layers || block_shape[1] != context_shape[0])
    {
        throw std::runtime_error(std::string(caller) + ": block_tables must be [num_layers, num_seqs, max_blocks_per_seq].");
    }
    const c10::Device device = inputs.input_ids.device();
    if (inputs.positions.device() != device
        || inputs.slot_mapping.device() != device
        || inputs.seq_indices.device() != device
        || inputs.context_lens.device() != device
        || inputs.block_tables.device() != device)
    {
        throw std::runtime_error(std::string(caller) + ": prepared tensors must be on the same device.");
    }
    if (input_shape[0] == 0)
    {
        return;
    }
    if (device.is_cuda())
    {
        return;
    }

    const Tensor input_cpu = tensor_to_cpu_contiguous(inputs.input_ids);
    const Tensor positions_cpu = tensor_to_cpu_contiguous(inputs.positions);
    const int32_t* input_ptr = input_cpu.data_ptr<int32_t>();
    const int32_t* position_ptr = positions_cpu.data_ptr<int32_t>();
    for (int64_t row = 0; row < input_shape[0]; ++row)
    {
        const int32_t token_id = input_ptr[row];
        if (token_id < 0 || token_id >= vocab_size)
        {
            throw std::runtime_error(std::string(caller) + ": token is out of model vocab range.");
        }
        if (position_ptr[row] < 0)
        {
            throw std::runtime_error(std::string(caller) + ": positions must be non-negative.");
        }
    }
}

} // namespace

ModelRunner::ModelRunner(Model* model, ExecutionContext* ctx, KVCache* kv)
    : model_(model), kv_(kv)
{
    if (kv_ != nullptr)
    {
        kv_block_size_tokens_ = kv_->block_size_tokens();
    }
    set_global_execution_context(ctx);
}

ModelRunner::ModelRunner(const EngineArgs& args, KVCache* kv)
    : kv_(kv)
{
    init_from_args(args);
}

ModelRunner::~ModelRunner()
{
    reset_global_execution_context();
}

void ModelRunner::init_from_args(const EngineArgs& args)
{
    owned_hf_loader_.reset();
    owned_hf_loaders_.clear();
    args.parallel_config.validate();

    if (args.kv_block_size_tokens <= 0)
    {
        throw std::runtime_error("ModelRunner: kv_block_size_tokens must be positive.");
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
                    throw std::runtime_error("ModelRunner: tiny_lm_checkpoint_path must be provided.");
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
                    throw std::runtime_error("ModelRunner: hf_model_dir must be provided.");
                }
                const LlamaConfig hf_config = HFLlamaConfigLoader::load_from_dir(args.hf_model_dir);
                WeightMap weight_map = load_weight_map_from_safetensors(
                    resolve_hf_safetensor_paths(args),
                    args.parallel_config,
                    owned_hf_loaders_);
                auto llama_model = std::make_unique<LlamaForCausalLM>(hf_config, std::move(weight_map));
                llama_model->allocate_buffers(resolve_model_max_batch_size(args), args.parallel_config);
                owned_model_ = std::move(llama_model);
                break;
            }
            case EngineModelType::kPrebuilt:
            default:
                throw std::runtime_error("ModelRunner: model pointer is null and no constructible model_type is configured.");
        }
        model_ = owned_model_.get();
    }

    initialize_global_execution_context(args, kv_);
}

int32_t ModelRunner::resolve_model_max_batch_size(const EngineArgs& args) const
{
    int32_t max_batch_size = args.max_batch_size;
    if (args.scheduler_config.max_prefill_tokens_per_step > max_batch_size)
    {
        max_batch_size = args.scheduler_config.max_prefill_tokens_per_step;
    }
    if (max_batch_size <= 0)
    {
        throw std::runtime_error("ModelRunner: resolved model max_batch_size must be positive.");
    }
    return max_batch_size;
}

void ModelRunner::validate_handles() const
{
    if (model_ == nullptr || g_execution_context == nullptr)
    {
        throw std::runtime_error("ModelRunner: model/context must be non-null.");
    }
}

int32_t ModelRunner::vocab_size() const
{
    validate_handles();
    return model_->vocab_size();
}

PreparedInputs ModelRunner::prepare_inputs(const SchedulerOutput& output)
{
    validate_handles();
    ExecutionContext& ctx = require_global_execution_context("ModelRunner::prepare_inputs");
    const c10::Device runtime_device = ctx.device();

    PreparedInputs prepared;
    prepared_req_ids_.clear();

    const int64_t request_count = static_cast<int64_t>(output.scheduled_reqs.size());
    const int64_t total_tokens = static_cast<int64_t>(std::max(0, output.total_num_scheduled_tokens));
    if (request_count == 0 || total_tokens == 0)
    {
        prepared.input_ids = make_int32_tensor_from_host({}, {0}, runtime_device, "ModelRunner::prepare_inputs");
        prepared.positions = make_int32_tensor_from_host({}, {0}, runtime_device, "ModelRunner::prepare_inputs");
        prepared.slot_mapping = make_int32_tensor_from_host({}, {0}, runtime_device, "ModelRunner::prepare_inputs");
        prepared.seq_indices = make_int32_tensor_from_host({}, {0}, runtime_device, "ModelRunner::prepare_inputs");
        prepared.context_lens = make_int32_tensor_from_host({}, {0}, runtime_device, "ModelRunner::prepare_inputs");
        prepared.block_tables = make_int32_tensor_from_host({}, {model_->num_layers(), 0, 0}, runtime_device, "ModelRunner::prepare_inputs");
        return prepared;
    }

    int64_t max_blocks_per_seq = 0;
    int64_t checked_total_tokens = 0;
    for (const RequestData& req_data : output.scheduled_reqs)
    {
        const auto count_it = output.num_scheduled_tokens.find(req_data.req_id);
        if (count_it == output.num_scheduled_tokens.end())
        {
            throw std::runtime_error("ModelRunner::prepare_inputs: missing token budget for scheduled request.");
        }
        const int32_t scheduled_tokens = count_it->second;
        if (scheduled_tokens <= 0)
        {
            throw std::runtime_error("ModelRunner::prepare_inputs: scheduled token budget must be positive.");
        }
        if (req_data.num_computed_tokens < 0)
        {
            throw std::runtime_error("ModelRunner::prepare_inputs: num_computed_tokens must be non-negative.");
        }
        if (req_data.new_token_ids.size() < static_cast<size_t>(scheduled_tokens))
        {
            throw std::runtime_error("ModelRunner::prepare_inputs: new_token_ids is shorter than scheduled budget.");
        }
        if (static_cast<int32_t>(req_data.block_tables.size()) != model_->num_layers())
        {
            throw std::runtime_error("ModelRunner::prepare_inputs: block_tables layer count must match model.");
        }
        for (const std::vector<int32_t>& layer_blocks : req_data.block_tables)
        {
            if (layer_blocks.empty())
            {
                throw std::runtime_error("ModelRunner::prepare_inputs: each layer block table must be non-empty.");
            }
            max_blocks_per_seq = std::max(max_blocks_per_seq, static_cast<int64_t>(layer_blocks.size()));
        }
        checked_total_tokens += scheduled_tokens;
    }
    if (checked_total_tokens != total_tokens)
    {
        throw std::runtime_error("ModelRunner::prepare_inputs: total scheduled token count mismatch.");
    }

    std::vector<int32_t> input_values(static_cast<size_t>(total_tokens), 0);
    std::vector<int32_t> position_values(static_cast<size_t>(total_tokens), 0);
    std::vector<int32_t> slot_values(static_cast<size_t>(total_tokens), 0);
    std::vector<int32_t> seq_index_values(static_cast<size_t>(total_tokens), 0);
    std::vector<int32_t> context_values(static_cast<size_t>(request_count), 0);
    std::vector<int32_t> block_table_values(
        static_cast<size_t>(model_->num_layers() * request_count * max_blocks_per_seq),
        -1);

    int64_t flat_token_index = 0;
    int64_t seq_index = 0;
    prepared.sample_row_offsets.reserve(static_cast<size_t>(request_count));
    prepared_req_ids_.reserve(static_cast<size_t>(request_count));

    for (const RequestData& req_data : output.scheduled_reqs)
    {
        const int32_t scheduled_tokens = output.num_scheduled_tokens.at(req_data.req_id);
        const int32_t context_len = req_data.num_computed_tokens + scheduled_tokens;
        context_values[static_cast<size_t>(seq_index)] = context_len;
        for (size_t layer = 0; layer < req_data.block_tables.size(); ++layer)
        {
            const std::vector<int32_t>& layer_blocks = req_data.block_tables[layer];
            for (size_t col = 0; col < layer_blocks.size(); ++col)
            {
                block_table_values[
                    static_cast<int64_t>(layer) * request_count * max_blocks_per_seq
                    + seq_index * max_blocks_per_seq
                    + static_cast<int64_t>(col)] = layer_blocks[col];
            }
        }

        for (int32_t i = 0; i < scheduled_tokens; ++i)
        {
            const int32_t position = req_data.num_computed_tokens + i;
            const int32_t logical_block_index = position / kv_block_size_tokens_;
            if (logical_block_index < 0
                || logical_block_index >= static_cast<int32_t>(req_data.block_tables[0].size()))
            {
                throw std::runtime_error("ModelRunner::prepare_inputs: logical block index is out of range.");
            }
            const int32_t physical_block_id = req_data.block_tables[0][static_cast<size_t>(logical_block_index)];
            if (physical_block_id < 0)
            {
                throw std::runtime_error("ModelRunner::prepare_inputs: physical block id must be non-negative.");
            }
            input_values[static_cast<size_t>(flat_token_index)] = req_data.new_token_ids[static_cast<size_t>(i)];
            position_values[static_cast<size_t>(flat_token_index)] = position;
            slot_values[static_cast<size_t>(flat_token_index)] =
                physical_block_id * kv_block_size_tokens_ + (position % kv_block_size_tokens_);
            seq_index_values[static_cast<size_t>(flat_token_index)] = static_cast<int32_t>(seq_index);
            ++flat_token_index;
        }

        prepared.sample_row_offsets.push_back(static_cast<int32_t>(flat_token_index - 1));
        prepared_req_ids_.push_back(req_data.req_id);
        ++seq_index;
    }

    if (flat_token_index != total_tokens)
    {
        throw std::runtime_error("ModelRunner::prepare_inputs: flattened token count mismatch.");
    }

    prepared.input_ids = make_int32_tensor_from_host(input_values, {total_tokens}, runtime_device, "ModelRunner::prepare_inputs");
    prepared.positions = make_int32_tensor_from_host(position_values, {total_tokens}, runtime_device, "ModelRunner::prepare_inputs");
    prepared.slot_mapping = make_int32_tensor_from_host(slot_values, {total_tokens}, runtime_device, "ModelRunner::prepare_inputs");
    prepared.seq_indices = make_int32_tensor_from_host(seq_index_values, {total_tokens}, runtime_device, "ModelRunner::prepare_inputs");
    prepared.context_lens = make_int32_tensor_from_host(context_values, {request_count}, runtime_device, "ModelRunner::prepare_inputs");
    prepared.block_tables = make_int32_tensor_from_host(
        block_table_values,
        {model_->num_layers(), request_count, max_blocks_per_seq},
        runtime_device,
        "ModelRunner::prepare_inputs");
    return prepared;
}

Tensor ModelRunner::run_model(const PreparedInputs& inputs, RuntimeProfilingStats* profiling) const
{
    validate_handles();
    validate_prepared_tensor_pack(inputs, model_->vocab_size(), model_->num_layers(), "ModelRunner::run_model");
    if (inputs.input_ids.numel() == 0)
    {
        return torch::empty(
            {0, model_->vocab_size()},
            torch::TensorOptions().dtype(to_torch_scalar_type(DType::kFloat32)).device(g_execution_context->device()));
    }

    ops::PagedAttentionRuntimeMetadata metadata;
    metadata.slot_mapping = &inputs.slot_mapping;
    metadata.seq_indices = &inputs.seq_indices;
    metadata.context_lens = &inputs.context_lens;
    metadata.block_tables = &inputs.block_tables;
    metadata.block_size_tokens = kv_block_size_tokens_;
    metadata.enabled = true;

    ExecutionContext& exec_ctx = require_global_execution_context("ModelRunner::run_model");
    RuntimeContext runtime_ctx(exec_ctx, metadata, profiling, runtime_detail_profile_enabled());
    auto guard = exec_ctx.step_guard();
    (void)guard;
    return model_->forward(inputs, runtime_ctx);
}

ModelRunnerOutput ModelRunner::run(const SchedulerOutput& scheduler_output)
{
    validate_handles();
    const c10::Device runtime_device = require_global_execution_context("ModelRunner::run").device();

    int64_t prefill_tokens = 0;
    int64_t decode_tokens = 0;
    for (const RequestData& req_data : scheduler_output.scheduled_reqs)
    {
        const auto count_it = scheduler_output.num_scheduled_tokens.find(req_data.req_id);
        if (count_it == scheduler_output.num_scheduled_tokens.end() || count_it->second <= 0)
        {
            continue;
        }
        if (req_data.is_prefill)
        {
            prefill_tokens += count_it->second;
        }
        else
        {
            decode_tokens += count_it->second;
        }
    }

    synchronize_for_profile(runtime_device);
    const auto prepare_start = ProfileClock::now();
    PreparedInputs inputs = prepare_inputs(scheduler_output);
    synchronize_for_profile(runtime_device);
    const auto prepare_end = ProfileClock::now();

    ModelRunnerOutput output;
    output.profiling.prepare_inputs_ms = elapsed_profile_ms(prepare_start, prepare_end);
    output.profiling.prefill_tokens = prefill_tokens;
    output.profiling.decode_tokens = decode_tokens;
    output.req_ids.reserve(prepared_req_ids_.size());
    output.sampled_token_ids.reserve(prepared_req_ids_.size());
    output.req_id_to_index.reserve(prepared_req_ids_.size());
    if (inputs.input_ids.numel() == 0)
    {
        return output;
    }

    synchronize_for_profile(runtime_device);
    const auto model_start = ProfileClock::now();
    Tensor logits = run_model(inputs, &output.profiling);
    synchronize_for_profile(runtime_device);
    const auto model_end = ProfileClock::now();

    const double model_ms = elapsed_profile_ms(model_start, model_end);
    const int64_t model_tokens = prefill_tokens + decode_tokens;
    if (model_tokens > 0 && prefill_tokens > 0 && decode_tokens > 0)
    {
        output.profiling.prefill_ms = model_ms * static_cast<double>(prefill_tokens) / static_cast<double>(model_tokens);
        output.profiling.decode_ms_total = model_ms * static_cast<double>(decode_tokens) / static_cast<double>(model_tokens);
    }
    else if (prefill_tokens > 0)
    {
        output.profiling.prefill_ms = model_ms;
    }
    else
    {
        output.profiling.decode_ms_total = model_ms;
    }

    synchronize_for_profile(runtime_device);
    const auto sampling_start = ProfileClock::now();
    std::vector<int32_t> sampled_rows = sample_greedy_rows(
        logits,
        inputs.sample_row_offsets,
        model_->vocab_size());
    synchronize_for_profile(runtime_device);
    const auto sampling_end = ProfileClock::now();
    output.profiling.sampling_ms = elapsed_profile_ms(sampling_start, sampling_end);
    output.profiling.sampled_tokens = static_cast<int64_t>(inputs.sample_row_offsets.size());

    for (size_t i = 0; i < prepared_req_ids_.size(); ++i)
    {
        const uint64_t req_id = prepared_req_ids_[i];
        const int32_t row = inputs.sample_row_offsets[i];
        if (row < 0 || static_cast<size_t>(row) >= sampled_rows.size())
        {
            throw std::runtime_error("ModelRunner::run: sampled row is out of range.");
        }
        const int32_t output_index = static_cast<int32_t>(output.req_ids.size());
        output.req_ids.push_back(req_id);
        output.sampled_token_ids.push_back(sampled_rows[static_cast<size_t>(row)]);
        output.req_id_to_index[req_id] = output_index;
    }

    return output;
}

} // namespace tiny_llm
