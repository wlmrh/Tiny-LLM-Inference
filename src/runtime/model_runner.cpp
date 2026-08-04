#include "tiny_llm/runtime/model_runner.h"

#include "tiny_llm/core/allocator.h"
#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/models/hf_llama_config_loader.h"
#include "tiny_llm/models/hf_safetensors_loader.h"
#include "tiny_llm/models/llama_model.h"
#include "tiny_llm/models/llama_weight_map.h"
#include "tiny_llm/models/model.h"
#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/kv_cache.h"
#include "tiny_llm/runtime/runtime_context.h"
#include "tiny_llm/runtime/sampler.h"
#include "tiny_llm/runtime/scheduler.h"

#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace tiny_llm
{

namespace
{

using ProfileClock = std::chrono::steady_clock;

double elapsed_profile_ms(ProfileClock::time_point start, ProfileClock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void synchronize_for_profile(const ExecutionContext &ctx)
{
#if TINYLLM_ENABLE_CUDA
    if (ctx.device().is_cuda())
    {
        const int device_index = ctx.device().index() >= 0 ? static_cast<int>(ctx.device().index()) : 0;
        cudaSetDevice(device_index);
        cudaStreamSynchronize(ctx.stream());
    }
#else
    (void)ctx;
#endif
}

bool runtime_detail_profile_enabled()
{
    const char *value = std::getenv("TINYLLM_PROFILE_DETAIL");
    if (value == nullptr)
    {
        return false;
    }
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

int32_t debug_logits_top_k()
{
    const char *value = std::getenv("TINYLLM_DEBUG_LOGITS_TOP_K");
    if (value == nullptr)
    {
        return 0;
    }
    try
    {
        const int parsed = std::stoi(value);
        return parsed > 0 ? parsed : 0;
    }
    catch (const std::exception &)
    {
        return 0;
    }
}

int64_t checked_numel(const std::vector<int64_t> &shape, const char *caller)
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

Tensor make_int32_tensor_from_host(const std::vector<int32_t> &values, const std::vector<int64_t> &shape,
                                   const c10::Device &device, const char *caller)
{
    const int64_t expected_numel = checked_numel(shape, caller);
    if (expected_numel != static_cast<int64_t>(values.size()))
    {
        throw std::runtime_error(std::string(caller) + ": host value count does not match tensor shape.");
    }

    Tensor cpu_tensor =
        torch::empty(shape, torch::TensorOptions().dtype(to_torch_scalar_type(DType::kInt32)).device(c10::kCPU));
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

struct PreparedRequestInfo
{
    // Non-owning scheduler record for this request; valid for the duration of prepare_batch().
    const RequestData *req_data = nullptr;
    // Consecutive tokens assigned to this request in the current scheduler step.
    int32_t scheduled_tokens = 0;
    // Total valid sequence length after the scheduled tokens are written to KV cache.
    int32_t context_len = 0;
    // Number of logical KV blocks needed to cover context_len.
    int32_t required_blocks = 0;
};

void populate_query_segments(const std::vector<PreparedRequestInfo> &request_infos, int64_t total_tokens,
                             PreparedInputs &prepared)
{
    prepared.query_segments.clear();
    prepared.query_segments_valid = false;
    if (request_infos.empty())
    {
        return;
    }

    int64_t row_start = 0;
    int32_t seq_index = 0;
    prepared.query_segments.reserve(request_infos.size());
    for (const PreparedRequestInfo &info : request_infos)
    {
        const RequestData &req_data = *info.req_data;
        prepared.query_segments.push_back(ops::PagedAttentionQuerySegment{
            row_start, seq_index, req_data.num_computed_tokens, info.scheduled_tokens});
        row_start += info.scheduled_tokens;
        ++seq_index;
    }

    if (row_start == total_tokens)
    {
        prepared.query_segments_valid = !prepared.query_segments.empty();
        return;
    }

    prepared.query_segments.clear();
}

int32_t required_block_count(int32_t context_len, int32_t block_size_tokens)
{
    if (context_len <= 0)
    {
        return 0;
    }
    return (context_len - 1) / block_size_tokens + 1;
}

int32_t checked_context_len(int32_t num_computed_tokens, int32_t scheduled_tokens, const char *caller)
{
    const int64_t context_len = static_cast<int64_t>(num_computed_tokens) + static_cast<int64_t>(scheduled_tokens);
    if (context_len > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
    {
        throw std::runtime_error(std::string(caller) + ": context length exceeds int32 range.");
    }
    return static_cast<int32_t>(context_len);
}

void validate_required_blocks(const RequestData &req_data, int32_t num_layers, int32_t required_blocks,
                              const char *caller)
{
    if (static_cast<int32_t>(req_data.block_tables.size()) != num_layers)
    {
        throw std::runtime_error(std::string(caller) + ": block_tables layer count must match model.");
    }
    for (const std::vector<int32_t> &layer_blocks : req_data.block_tables)
    {
        if (static_cast<int32_t>(layer_blocks.size()) < required_blocks)
        {
            throw std::runtime_error(std::string(caller) + ": block table does not cover scheduled context.");
        }
        for (int32_t block_index = 0; block_index < required_blocks; ++block_index)
        {
            if (layer_blocks[static_cast<size_t>(block_index)] < 0)
            {
                throw std::runtime_error(std::string(caller) + ": physical block id must be non-negative.");
            }
        }
    }
}

Tensor tensor_to_cpu_contiguous(const Tensor &tensor)
{
    if (tensor.device().is_cpu())
    {
        return tensor.contiguous();
    }
    return tensor.to(c10::kCPU, /*non_blocking=*/false, /*copy=*/true).contiguous();
}

std::vector<int32_t> top_k_token_ids(const float *logits, int32_t vocab_size, int32_t k)
{
    std::vector<int32_t> top_tokens;
    top_tokens.reserve(static_cast<size_t>(std::min(k, vocab_size)));
    for (int32_t token = 0; token < vocab_size; ++token)
    {
        const float value = logits[token];
        auto insert_pos = top_tokens.begin();
        while (insert_pos != top_tokens.end() && logits[*insert_pos] >= value)
        {
            ++insert_pos;
        }
        if (static_cast<int32_t>(top_tokens.size()) < k)
        {
            top_tokens.insert(insert_pos, token);
        }
        else if (insert_pos != top_tokens.end())
        {
            top_tokens.insert(insert_pos, token);
            top_tokens.pop_back();
        }
    }
    return top_tokens;
}

bool token_in_history(int32_t token_id, const std::vector<int32_t> &history)
{
    return std::find(history.begin(), history.end(), token_id) != history.end();
}

std::vector<std::filesystem::path> resolve_hf_safetensor_paths(const EngineArgs &args)
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
        throw std::runtime_error("ModelRunner: hf_model_dir does not exist or is not a directory: " +
                                 args.hf_model_dir);
    }

    for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(model_dir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        const std::filesystem::path &path = entry.path();
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

WeightMap load_weight_map_from_safetensors(const std::vector<std::filesystem::path> &paths,
                                           const ParallelConfig &parallel_config,
                                           std::vector<std::unique_ptr<HFSafeTensorLoader>> &owned_loaders)
{
    WeightMap weight_map;
    parallel_config.validate();
    const c10::Device target_device = parallel_config.torch_device();
    owned_loaders.clear();
    owned_loaders.reserve(paths.size());

    for (const std::filesystem::path &path : paths)
    {
        auto loader = std::make_unique<HFSafeTensorLoader>(HFSafeTensorLoader::from_file(path.string()));
        for (const std::string &key : loader->keys())
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

void validate_prepared_tensor_pack(const PreparedInputs &inputs, int32_t num_layers, const char *caller)
{
    if (!inputs.input_ids.defined() || !inputs.positions.defined() || !inputs.slot_mapping.defined() ||
        !inputs.seq_indices.defined() || !inputs.context_lens.defined() || !inputs.block_tables.defined())
    {
        throw std::runtime_error(std::string(caller) + ": all prepared tensors must be defined.");
    }
    if (tensor_dtype(inputs.input_ids) != DType::kInt32 || tensor_dtype(inputs.positions) != DType::kInt32 ||
        tensor_dtype(inputs.slot_mapping) != DType::kInt32 || tensor_dtype(inputs.seq_indices) != DType::kInt32 ||
        tensor_dtype(inputs.context_lens) != DType::kInt32 || tensor_dtype(inputs.block_tables) != DType::kInt32)
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
        throw std::runtime_error(std::string(caller) +
                                 ": block_tables must be [num_layers, num_seqs, max_blocks_per_seq].");
    }
    const c10::Device device = inputs.input_ids.device();
    if (inputs.positions.device() != device || inputs.slot_mapping.device() != device ||
        inputs.seq_indices.device() != device || inputs.context_lens.device() != device ||
        inputs.block_tables.device() != device)
    {
        throw std::runtime_error(std::string(caller) + ": prepared tensors must be on the same device.");
    }
}

} // namespace

ModelRunner::ModelRunner(const EngineArgs &args, KVCache *kv) : kv_(kv)
{
    init_from_args(args);
}

ModelRunner::~ModelRunner() = default;

void ModelRunner::init_from_args(const EngineArgs &args)
{
    owned_hf_loaders_.clear();
    args.parallel_config.validate();
    if (args.compute_dtype == RuntimeDType::kBFloat16 && !args.parallel_config.is_cuda())
    {
        throw std::runtime_error("ModelRunner: bfloat16 compute requires a CUDA device.");
    }
    if (args.kv_cache_dtype == RuntimeDType::kBFloat16 && !args.parallel_config.is_cuda())
    {
        throw std::runtime_error("ModelRunner: bfloat16 KV cache requires a CUDA device.");
    }

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
        case EngineModelType::kHFLlamaSafeTensor:
        {
            if (args.hf_model_dir.empty())
            {
                throw std::runtime_error("ModelRunner: hf_model_dir must be provided.");
            }
            const LlamaConfig hf_config = HFLlamaConfigLoader::load_from_dir(args.hf_model_dir);
            WeightMap weight_map = load_weight_map_from_safetensors(resolve_hf_safetensor_paths(args),
                                                                    args.parallel_config, owned_hf_loaders_);
            auto llama_model = std::make_unique<LlamaForCausalLM>(hf_config, std::move(weight_map));
            llama_model->allocate_buffers(resolve_model_max_batch_size(args), args.parallel_config);
            owned_model_ = std::move(llama_model);
            break;
        }
        case EngineModelType::kPrebuilt:
        default:
            throw std::runtime_error(
                "ModelRunner: model pointer is null and no constructible model_type is configured.");
        }
        model_ = owned_model_.get();
    }

    if (args.ctx != nullptr)
    {
        execution_context_ = args.ctx;
    }
    else
    {
        if (args.workspace != nullptr && args.workspace->parallel_config() != args.parallel_config)
        {
            throw std::runtime_error("ModelRunner: workspace device does not match parallel_config.");
        }
        owned_execution_context_ = std::make_unique<ExecutionContext>(args.execution_stream, args.workspace, kv_,
                                                                      args.parallel_config, args.compute_dtype);
        execution_context_ = owned_execution_context_.get();
    }
    if (execution_context_->parallel_config() != args.parallel_config)
    {
        throw std::runtime_error("ModelRunner: execution context device does not match parallel_config.");
    }
    if (execution_context_->compute_dtype() != args.compute_dtype)
    {
        throw std::runtime_error("ModelRunner: execution context compute dtype does not match EngineArgs.");
    }
}

int32_t ModelRunner::resolve_model_max_batch_size(const EngineArgs &args) const
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
    if (model_ == nullptr)
    {
        throw std::runtime_error("ModelRunner: model must be non-null.");
    }
    if (execution_context_ == nullptr)
    {
        throw std::runtime_error("ModelRunner: execution context must be non-null.");
    }
}

void ModelRunner::validate_token_ids(const std::vector<int32_t> &token_ids, const char *context) const
{
    validate_handles();
    const int32_t vocab_size = model_->vocab_size();
    for (int32_t token_id : token_ids)
    {
        if (token_id < 0 || token_id >= vocab_size)
        {
            const char *prefix = context != nullptr ? context : "ModelRunner::validate_token_ids";
            throw std::runtime_error(std::string(prefix) + ": token is out of model vocab range.");
        }
    }
}

ModelRunner::PreparedBatch ModelRunner::prepare_batch(const SchedulerOutput &output, ExecutionContext &ctx)
{
    constexpr const char *kCaller = "ModelRunner::prepare_batch";
    const c10::Device runtime_device = ctx.device();

    PreparedBatch batch;
    PreparedInputs &prepared = batch.inputs;

    const int64_t request_count = static_cast<int64_t>(output.scheduled_reqs.size());
    batch.scheduling_stats.scheduled_requests = request_count;
    const int64_t total_tokens = static_cast<int64_t>(std::max(0, output.total_num_scheduled_tokens));
    if (request_count == 0 || total_tokens == 0)
    {
        prepared.input_ids = make_int32_tensor_from_host({}, {0}, runtime_device, kCaller);
        prepared.positions = make_int32_tensor_from_host({}, {0}, runtime_device, kCaller);
        prepared.slot_mapping = make_int32_tensor_from_host({}, {0}, runtime_device, kCaller);
        prepared.seq_indices = make_int32_tensor_from_host({}, {0}, runtime_device, kCaller);
        prepared.context_lens = make_int32_tensor_from_host({}, {0}, runtime_device, kCaller);
        prepared.block_tables = make_int32_tensor_from_host({}, {model_->num_layers(), 0, 0}, runtime_device, kCaller);
        return batch;
    }

    const int32_t num_layers = model_->num_layers();
    const int32_t vocab_size = model_->vocab_size();
    std::vector<PreparedRequestInfo> request_infos;
    request_infos.reserve(static_cast<size_t>(request_count));
    int64_t max_blocks_per_seq = 0;
    int64_t checked_total_tokens = 0;
    for (const RequestData &req_data : output.scheduled_reqs)
    {
        const auto count_it = output.num_scheduled_tokens.find(req_data.req_id);
        if (count_it == output.num_scheduled_tokens.end())
        {
            throw std::runtime_error(std::string(kCaller) + ": missing token budget for scheduled request.");
        }
        const int32_t scheduled_tokens = count_it->second;
        if (scheduled_tokens <= 0)
        {
            throw std::runtime_error(std::string(kCaller) + ": scheduled token budget must be positive.");
        }
        if (req_data.num_computed_tokens < 0)
        {
            throw std::runtime_error(std::string(kCaller) + ": num_computed_tokens must be non-negative.");
        }
        if (req_data.new_token_ids.size() < static_cast<size_t>(scheduled_tokens))
        {
            throw std::runtime_error(std::string(kCaller) + ": new_token_ids is shorter than scheduled budget.");
        }
        const int32_t context_len = checked_context_len(req_data.num_computed_tokens, scheduled_tokens, kCaller);
        const int32_t required_blocks = required_block_count(context_len, kv_block_size_tokens_);
        validate_required_blocks(req_data, num_layers, required_blocks, kCaller);

        for (const std::vector<int32_t> &layer_blocks : req_data.block_tables)
        {
            max_blocks_per_seq = std::max(max_blocks_per_seq, static_cast<int64_t>(layer_blocks.size()));
        }

        bool request_has_prefill = false;
        bool request_has_decode = false;
        batch.scheduling_stats.max_context_len =
            std::max<int64_t>(batch.scheduling_stats.max_context_len, static_cast<int64_t>(context_len));
        for (int32_t i = 0; i < scheduled_tokens; ++i)
        {
            const int32_t position = req_data.num_computed_tokens + i;
            if (position < req_data.prompt_token_count)
            {
                ++batch.scheduling_stats.prefill_tokens;
                request_has_prefill = true;
            }
            else
            {
                ++batch.scheduling_stats.decode_tokens;
                request_has_decode = true;
            }
        }
        if (request_has_prefill)
        {
            ++batch.scheduling_stats.prefill_requests;
        }
        if (request_has_decode)
        {
            ++batch.scheduling_stats.decode_requests;
        }

        request_infos.push_back(PreparedRequestInfo{&req_data, scheduled_tokens, context_len, required_blocks});
        checked_total_tokens += scheduled_tokens;
    }
    if (checked_total_tokens != total_tokens)
    {
        throw std::runtime_error(std::string(kCaller) + ": total scheduled token count mismatch.");
    }
    batch.scheduling_stats.scheduled_tokens =
        batch.scheduling_stats.prefill_tokens + batch.scheduling_stats.decode_tokens;
    batch.scheduling_stats.profiled_steps = batch.scheduling_stats.scheduled_tokens > 0 ? 1 : 0;

    // Scheduled token IDs flattened in request-major order, shape [total_tokens].
    std::vector<int32_t> input_values(static_cast<size_t>(total_tokens), 0);
    // Absolute zero-based token positions within each request, shape [total_tokens].
    std::vector<int32_t> position_values(static_cast<size_t>(total_tokens), 0);
    // KV slot indices derived from the layer-0 block table, shape [total_tokens].
    std::vector<int32_t> slot_values(static_cast<size_t>(total_tokens), 0);
    // Batch-local request index for each flattened token, shape [total_tokens].
    std::vector<int32_t> seq_index_values(static_cast<size_t>(total_tokens), 0);
    // Computed context length after this step for each request, shape [request_count].
    std::vector<int32_t> context_values(static_cast<size_t>(request_count), 0);
    // Dense [num_layers, request_count, max_blocks_per_seq] block table, padded with -1.
    std::vector<int32_t> block_table_values(static_cast<size_t>(num_layers * request_count * max_blocks_per_seq), -1);

    int64_t flat_token_index = 0;
    int64_t seq_index = 0;
    prepared.sample_row_offsets.reserve(static_cast<size_t>(request_count));
    batch.req_ids.reserve(static_cast<size_t>(request_count));
    batch.sampling_params.reserve(static_cast<size_t>(request_count));
    batch.token_histories.reserve(static_cast<size_t>(request_count));

    for (const PreparedRequestInfo &info : request_infos)
    {
        const RequestData &req_data = *info.req_data;
        context_values[static_cast<size_t>(seq_index)] = info.context_len;
        for (size_t layer = 0; layer < req_data.block_tables.size(); ++layer)
        {
            const std::vector<int32_t> &layer_blocks = req_data.block_tables[layer];
            for (size_t col = 0; col < layer_blocks.size(); ++col)
            { // block_table_values[num_layers, num_seqs, max_blocks_per_seq]
                block_table_values[static_cast<int64_t>(layer) * request_count * max_blocks_per_seq +
                                   seq_index * max_blocks_per_seq + static_cast<int64_t>(col)] = layer_blocks[col];
            }
        }

        for (int32_t i = 0; i < info.scheduled_tokens; ++i)
        {
            // Absolute token position within this request.
            const int32_t position = req_data.num_computed_tokens + i;
            const int32_t logical_block_index = position / kv_block_size_tokens_;
            const int32_t token_id = req_data.new_token_ids[static_cast<size_t>(i)];
            if (token_id < 0 || token_id >= vocab_size)
            {
                throw std::runtime_error(std::string(kCaller) + ": token is out of model vocab range.");
            }
            const int32_t physical_block_id = req_data.block_tables[0][static_cast<size_t>(logical_block_index)];
            input_values[static_cast<size_t>(flat_token_index)] = token_id;
            position_values[static_cast<size_t>(flat_token_index)] = position;
            slot_values[static_cast<size_t>(flat_token_index)] =
                physical_block_id * kv_block_size_tokens_ + (position % kv_block_size_tokens_);
            seq_index_values[static_cast<size_t>(flat_token_index)] = static_cast<int32_t>(seq_index);
            ++flat_token_index;
        }

        prepared.sample_row_offsets.push_back(static_cast<int32_t>(flat_token_index - 1));
        batch.req_ids.push_back(req_data.req_id);
        batch.sampling_params.push_back(req_data.sampling_params);
        batch.token_histories.push_back(req_data.context_token_ids);
        ++seq_index;
    }

    if (flat_token_index != total_tokens)
    {
        throw std::runtime_error(std::string(kCaller) + ": flattened token count mismatch.");
    }

    prepared.input_ids = make_int32_tensor_from_host(input_values, {total_tokens}, runtime_device, kCaller);
    prepared.positions = make_int32_tensor_from_host(position_values, {total_tokens}, runtime_device, kCaller);
    prepared.slot_mapping = make_int32_tensor_from_host(slot_values, {total_tokens}, runtime_device, kCaller);
    prepared.seq_indices = make_int32_tensor_from_host(seq_index_values, {total_tokens}, runtime_device, kCaller);
    prepared.context_lens = make_int32_tensor_from_host(context_values, {request_count}, runtime_device, kCaller);
    prepared.block_tables = make_int32_tensor_from_host(
        block_table_values, {num_layers, request_count, max_blocks_per_seq}, runtime_device, kCaller);
    prepared.host_block_tables = std::move(block_table_values);
    populate_query_segments(request_infos, total_tokens, prepared);
    return batch;
}

Tensor ModelRunner::run_model(const PreparedInputs &inputs, ExecutionContext &exec_ctx,
                              RuntimeProfilingStats *profiling) const
{
    validate_prepared_tensor_pack(inputs, model_->num_layers(), "ModelRunner::run_model");
    if (inputs.input_ids.numel() == 0)
    {
        return torch::empty(
            {0, model_->vocab_size()},
            torch::TensorOptions().dtype(to_torch_scalar_type(DType::kFloat32)).device(exec_ctx.device()));
    }

    ops::PagedAttentionRuntimeMetadata metadata;
    metadata.slot_mapping = &inputs.slot_mapping;
    metadata.seq_indices = &inputs.seq_indices;
    metadata.context_lens = &inputs.context_lens;
    metadata.block_tables = &inputs.block_tables;
    metadata.host_block_tables = inputs.host_block_tables.empty() ? nullptr : inputs.host_block_tables.data();
    metadata.host_block_table_count = static_cast<int64_t>(inputs.host_block_tables.size());
    metadata.query_segments = inputs.query_segments.empty() ? nullptr : inputs.query_segments.data();
    metadata.query_segment_count = static_cast<int64_t>(inputs.query_segments.size());
    metadata.block_size_tokens = kv_block_size_tokens_;
    metadata.query_segments_valid = inputs.query_segments_valid;
    metadata.enabled = true;

    RuntimeContext runtime_ctx(exec_ctx, metadata, profiling, runtime_detail_profile_enabled());
    auto guard = exec_ctx.step_guard();
    (void)guard;
    c10::InferenceMode inference_guard(true);
    return model_->forward(inputs, runtime_ctx);
}

ModelRunnerOutput ModelRunner::run(const SchedulerOutput &scheduler_output)
{
    if (model_ == nullptr)
    {
        throw std::runtime_error("ModelRunner: model must be non-null.");
    }
    validate_handles();
    ExecutionContext &exec_ctx = *execution_context_;
    synchronize_for_profile(exec_ctx);
    const auto prepare_start = ProfileClock::now();
    PreparedBatch batch = prepare_batch(scheduler_output, exec_ctx);
    const PreparedInputs &inputs = batch.inputs;
    synchronize_for_profile(exec_ctx);
    const auto prepare_end = ProfileClock::now();

    ModelRunnerOutput output;
    output.profiling = batch.scheduling_stats;
    output.profiling.prepare_inputs_ms = elapsed_profile_ms(prepare_start, prepare_end);
    output.sampled_token_ids.reserve(batch.req_ids.size());
    output.req_id_to_index.reserve(batch.req_ids.size());
    if (inputs.input_ids.numel() == 0)
    {
        debug_step_index_ = 0;
        return output;
    }

    synchronize_for_profile(exec_ctx);
    const auto model_start = ProfileClock::now();
    Tensor logits = run_model(inputs, exec_ctx, &output.profiling);
    synchronize_for_profile(exec_ctx);
    const auto model_end = ProfileClock::now();

    std::vector<int32_t> logit_sample_rows = inputs.sample_row_offsets;
    if (logits.size(0) == static_cast<int64_t>(inputs.sample_row_offsets.size()) &&
        logits.size(0) != inputs.input_ids.size(0))
    {
        logit_sample_rows.resize(inputs.sample_row_offsets.size());
        for (size_t i = 0; i < logit_sample_rows.size(); ++i)
        {
            logit_sample_rows[i] = static_cast<int32_t>(i);
        }
    }
    else if (logits.size(0) != inputs.input_ids.size(0))
    {
        throw std::runtime_error("ModelRunner::run: model logits rows must match either input rows or sample rows.");
    }

    const int32_t debug_top_k = debug_logits_top_k();
    if (debug_top_k > 0)
    {
        Tensor logits_cpu = tensor_to_cpu_contiguous(logits);
        Tensor positions_cpu = tensor_to_cpu_contiguous(inputs.positions);
        Tensor slot_mapping_cpu = tensor_to_cpu_contiguous(inputs.slot_mapping);
        Tensor seq_indices_cpu = tensor_to_cpu_contiguous(inputs.seq_indices);
        Tensor context_lens_cpu = tensor_to_cpu_contiguous(inputs.context_lens);
        Tensor block_tables_cpu = tensor_to_cpu_contiguous(inputs.block_tables);
        const float *logits_ptr = logits_cpu.data_ptr<float>();
        const int32_t *position_ptr = positions_cpu.data_ptr<int32_t>();
        const int32_t *slot_ptr = slot_mapping_cpu.data_ptr<int32_t>();
        const int32_t *seq_index_ptr = seq_indices_cpu.data_ptr<int32_t>();
        const int32_t *context_ptr = context_lens_cpu.data_ptr<int32_t>();
        const int32_t *block_ptr = block_tables_cpu.data_ptr<int32_t>();
        const std::vector<int64_t> block_shape = tensor_shape(block_tables_cpu);
        const int64_t num_seqs = block_shape.size() == 3 ? block_shape[1] : 0;
        const int64_t max_blocks_per_seq = block_shape.size() == 3 ? block_shape[2] : 0;

        for (size_t i = 0; i < batch.req_ids.size(); ++i)
        {
            const int32_t input_row = inputs.sample_row_offsets[i];
            const int32_t logit_row = logit_sample_rows[i];
            const int32_t seq_index = seq_index_ptr[input_row];
            const std::vector<int32_t> top_tokens =
                top_k_token_ids(logits_ptr + static_cast<size_t>(logit_row) * static_cast<size_t>(model_->vocab_size()),
                                model_->vocab_size(), debug_top_k);

            std::cerr << "{\"event\":\"tinyllm_model_runner_logits\",";
            std::cerr << "\"step\":" << debug_step_index_ << ",";
            std::cerr << "\"req_id\":" << batch.req_ids[i] << ",";
            std::cerr << "\"sample_index\":" << i << ",";
            std::cerr << "\"sample_row\":" << input_row << ",";
            std::cerr << "\"logit_row\":" << logit_row << ",";
            std::cerr << "\"position\":" << position_ptr[input_row] << ",";
            std::cerr << "\"slot\":" << slot_ptr[input_row] << ",";
            std::cerr << "\"seq_index\":" << seq_index << ",";
            std::cerr << "\"context_len\":"
                      << ((seq_index >= 0 && seq_index < context_lens_cpu.numel()) ? context_ptr[seq_index] : -1)
                      << ",";
            std::cerr << "\"repetition_penalty\":" << batch.sampling_params[i].repetition_penalty << ",";
            std::cerr << "\"top_tokens\":[";
            for (size_t j = 0; j < top_tokens.size(); ++j)
            {
                if (j != 0)
                    std::cerr << ",";
                std::cerr << top_tokens[j];
            }
            std::cerr << "],\"top_logits\":[";
            for (size_t j = 0; j < top_tokens.size(); ++j)
            {
                if (j != 0)
                    std::cerr << ",";
                std::cerr << logits_ptr[static_cast<size_t>(logit_row) * static_cast<size_t>(model_->vocab_size()) +
                                        static_cast<size_t>(top_tokens[j])];
            }
            std::cerr << "],\"top_in_history\":[";
            for (size_t j = 0; j < top_tokens.size(); ++j)
            {
                if (j != 0)
                    std::cerr << ",";
                std::cerr << (token_in_history(top_tokens[j], batch.token_histories[i]) ? "true" : "false");
            }
            std::cerr << "],\"layer0_blocks\":[";
            if (seq_index >= 0 && seq_index < num_seqs)
            {
                const int64_t block_limit = std::min<int64_t>(max_blocks_per_seq, 8);
                bool first = true;
                for (int64_t col = 0; col < block_limit; ++col)
                {
                    const int32_t block_id = block_ptr[static_cast<int64_t>(seq_index) * max_blocks_per_seq + col];
                    if (block_id < 0)
                        break;
                    if (!first)
                        std::cerr << ",";
                    first = false;
                    std::cerr << block_id;
                }
            }
            std::cerr << "]}\n";
        }
        ++debug_step_index_;
    }

    const double model_ms = elapsed_profile_ms(model_start, model_end);
    output.profiling.model_ms_total = model_ms;
    if (output.profiling.prefill_tokens > 0 && output.profiling.decode_tokens > 0)
    {
        output.profiling.mixed_model_ms = model_ms;
    }
    else if (output.profiling.prefill_tokens > 0)
    {
        output.profiling.prefill_ms = model_ms;
    }
    else if (output.profiling.decode_tokens > 0)
    {
        output.profiling.decode_ms_total = model_ms;
    }

    synchronize_for_profile(exec_ctx);
    const auto sampling_start = ProfileClock::now();
    const SamplerBatch sampler_batch{logit_sample_rows, model_->vocab_size(), &batch.token_histories,
                                     &batch.sampling_params, &batch.req_ids};
    std::vector<int32_t> sampled_rows = sample_rows(logits, sampler_batch);
    synchronize_for_profile(exec_ctx);
    const auto sampling_end = ProfileClock::now();
    output.profiling.sampling_ms = elapsed_profile_ms(sampling_start, sampling_end);
    output.profiling.sampled_tokens = static_cast<int64_t>(inputs.sample_row_offsets.size());

    for (size_t i = 0; i < batch.req_ids.size(); ++i)
    {
        const uint64_t req_id = batch.req_ids[i];
        const int32_t row = logit_sample_rows[i];
        if (row < 0 || static_cast<size_t>(row) >= sampled_rows.size())
        {
            throw std::runtime_error("ModelRunner::run: sampled row is out of range.");
        }
        const int32_t output_index = static_cast<int32_t>(output.sampled_token_ids.size());
        output.sampled_token_ids.push_back(sampled_rows[static_cast<size_t>(row)]);
        output.req_id_to_index[req_id] = output_index;
    }

    return output;
}

} // namespace tiny_llm
