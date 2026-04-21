#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/runtime/scheduler.h"

namespace tiny_llm {

class ExecutionContext;
class KVCache;
class Model;
struct EngineArgs;

/**
 * @brief Thin executor wrapper around single-device model forward and sampling.
 */
class ModelExecutor {
public:
    ModelExecutor(const EngineArgs& args, KVCache* kv);
    ModelExecutor(Model* model, ExecutionContext* ctx, KVCache* kv);
    ~ModelExecutor();

    int32_t vocab_size() const;

    ModelRunnerOutput execute_model(const SchedulerOutput& scheduler_output);

private:
    void init_from_args(const EngineArgs& args);
    void validate_handles() const;
    std::vector<int32_t> run_forward_batch(const Tensor& input_tokens,
                                           const Tensor& position_ids,
                                           const Tensor& slot_mapping,
                                           const Tensor& context_lens,
                                           const Tensor& block_tables,
                                           const std::vector<int32_t>& core_seq_ids,
                                           const std::vector<int32_t>& req_end_offsets,
                                           bool need_sampling) const;

    std::unique_ptr<Model> owned_model_;
    Model* model_ = nullptr;
    KVCache* kv_ = nullptr;
    int32_t kv_block_size_tokens_ = 16;
};

} // namespace tiny_llm
