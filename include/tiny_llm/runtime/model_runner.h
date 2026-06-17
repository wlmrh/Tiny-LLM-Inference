#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/model_runner_output.h"
#include "tiny_llm/runtime/prepared_inputs.h"
#include "tiny_llm/runtime/processors.h"

namespace tiny_llm {

class HFSafeTensorLoader;
class KVCache;
class Model;
struct EngineArgs;
struct SchedulerOutput;

/**
 * @brief Converts SchedulerOutput to model tensors, runs the model, and samples logits.
 */
class ModelRunner {
public:
    ModelRunner(const EngineArgs& args, KVCache* kv);
    ~ModelRunner();

    void validate_token_ids(const std::vector<int32_t>& token_ids, const char* context) const;
    ModelRunnerOutput run(const SchedulerOutput& scheduler_output);

private:
    struct PreparedBatch {
        PreparedInputs inputs;
        std::vector<uint64_t> req_ids;
        std::vector<SamplingParams> sampling_params;
        std::vector<std::vector<int32_t>> token_histories;
    };

    void init_from_args(const EngineArgs& args);
    void validate_handles() const;
    int32_t resolve_model_max_batch_size(const EngineArgs& args) const;
    PreparedBatch prepare_batch(const SchedulerOutput& scheduler_output);
    Tensor run_model(const PreparedInputs& inputs, RuntimeProfilingStats* profiling) const;

    std::unique_ptr<Model> owned_model_;
    std::vector<std::unique_ptr<HFSafeTensorLoader>> owned_hf_loaders_;
    Model* model_ = nullptr;
    KVCache* kv_ = nullptr;
    int32_t kv_block_size_tokens_ = 16;

    int64_t debug_step_index_ = 0;
};

} // namespace tiny_llm
