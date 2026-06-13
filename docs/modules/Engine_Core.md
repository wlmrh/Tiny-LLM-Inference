# Engine Core Module

`EngineCore` is the token-level orchestrator. It has no tokenizer or string output logic; it receives already-tokenized requests and returns sampled token IDs.

## Main Files

- `include/tiny_llm/runtime/engine_core.h`
- `src/runtime/engine_core.cpp`
- `include/tiny_llm/runtime/processors.h`
- `include/tiny_llm/runtime/scheduler.h`

## Responsibilities

- Own the scheduler and model runner.
- Validate prompt token IDs against the model vocabulary.
- Convert `EngineCoreRequest` into scheduler `Request`.
- Run one inference step by scheduling work, executing model forward/sampling, and updating scheduler state.
- Expose the most recent `RuntimeProfilingStats`.

## Attributes

- `scheduler_`: owned `Scheduler`, including request queues and KV cache manager.
- `runner_`: owned `ModelRunner`, bound to the scheduler-owned KV cache.
- `last_step_profile_`: profiling data from the most recent `ModelRunner::run`.

## Interfaces

- `EngineCore(const EngineArgs& args)`: constructs scheduler and model runner from shared runtime args.
- `add_request(const EngineCoreRequest& request)`: validates token range and enqueues a scheduler request.
- `step()`: runs one scheduler/model/update cycle and returns `(outputs, has_scheduled_tokens)`.
- `last_step_profile()`: returns profiling for the last call to `step()`.

## Step Semantics

`step()` performs:

1. Return an empty result if the scheduler has no unfinished requests.
2. Call `Scheduler::schedule()` to produce `SchedulerOutput`.
3. Call `ModelRunner::run(scheduler_output)`.
4. Store `model_output.profiling`.
5. Call `Scheduler::update_from_output(...)`.
6. Convert ordered `std::map<int, EngineCoreOutput>` into `std::unordered_map<int, EngineCoreOutput>`.

The boolean return value is true when the scheduler had at least one scheduled token in this step. This is used by `LLMEngine` to verify the cleanup step does not unexpectedly run more work.

## Request Conversion

`EngineCore::add_request()` builds a scheduler `Request` with:

- `request_id = EngineCoreRequest::internal_id`
- `prompt_token_ids = EngineCoreRequest::prompt_token_ids`
- `sampling_params = EngineCoreRequest::sampling_params`

The scheduler initializes `_all_token_ids`, `status`, and `num_computed` when the request is enqueued.
