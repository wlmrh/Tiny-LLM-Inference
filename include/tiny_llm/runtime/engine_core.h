#pragma once

#include <cstdint>
#include <memory>
#include <tuple>
#include <unordered_map>

#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/processors.h"
#include "tiny_llm/runtime/runtime_stats.h"

namespace tiny_llm {

class ModelRunner;
class Scheduler;

/**
 * @brief Inner runtime core: scheduler, KV binding and model forward.
 */
class EngineCore {
public:
    explicit EngineCore(const EngineArgs& args);

    ~EngineCore();

    void add_request(const EngineCoreRequest& request);

    std::tuple<std::unordered_map<int, EngineCoreOutput>, bool> step();
    const RuntimeProfilingStats& last_step_profile() const { return last_step_profile_; }

private:
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<ModelRunner> runner_;
    RuntimeProfilingStats last_step_profile_;
};

} // namespace tiny_llm
