#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/prepared_inputs.h"
#include "tiny_llm/runtime/sampler.h"
#include "tiny_llm/runtime/scheduler.h"

namespace tiny_llm {

class ExecutionContext;
class HFSafeTensorLoader;
class KVCache;
class Model;
struct EngineArgs;

/**
 * @brief Converts SchedulerOutput to model tensors, runs the model, and samples logits.
 */
class ModelRunner {
public:
    ModelRunner(const EngineArgs& args, KVCache* kv);
    ModelRunner(Model* model, ExecutionContext* ctx, KVCache* kv);
    ~ModelRunner();

    int32_t vocab_size() const;

    PreparedInputs prepare_inputs(const SchedulerOutput& scheduler_output);
    ModelRunnerOutput run(const SchedulerOutput& scheduler_output);

private:
    void init_from_args(const EngineArgs& args);
    void validate_handles() const;
    int32_t resolve_model_max_batch_size(const EngineArgs& args) const;
    Tensor run_model(const PreparedInputs& inputs) const;

    std::unique_ptr<Model> owned_model_;
    std::unique_ptr<HFSafeTensorLoader> owned_hf_loader_;
    std::vector<std::unique_ptr<HFSafeTensorLoader>> owned_hf_loaders_;
    Model* model_ = nullptr;
    KVCache* kv_ = nullptr;
    int32_t kv_block_size_tokens_ = 16;

    std::vector<uint64_t> prepared_req_ids_;
};

} // namespace tiny_llm
