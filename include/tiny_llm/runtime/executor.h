#pragma once

#include <cstdint>
#include <memory>
#include <optional>

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

    void execute_model(const SchedulerOutput& scheduler_output);
    ModelRunnerOutput sample_tokens(const SchedulerOutput& grammar_output);

private:
    void init_from_args(const EngineArgs& args);
    void validate_handles() const;
    void run_prefill_token(int32_t core_seq_id, int32_t token, int32_t position) const;
    int32_t run_decode_and_sample(int32_t core_seq_id, int32_t token, int32_t position) const;

    std::unique_ptr<Model> owned_model_;
    std::optional<ModelRunnerOutput> staged_model_output_;
    Model* model_ = nullptr;
    KVCache* kv_ = nullptr;
};

} // namespace tiny_llm
