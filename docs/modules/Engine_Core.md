# Engine Core Module

`EngineCore` is the token-level orchestrator below `LLMEngine`. It has no tokenizer or string output logic; it receives already-tokenized requests and returns sampled token IDs.

## Project Position

`EngineCore` sits between frontend request processing and the execution backend:

```text
LLMEngine
  -> EngineCore
       -> Scheduler
       -> ModelRunner
            -> Model
```

`LLMEngine` owns text/token conversion and user-facing output assembly. `EngineCore` owns the scheduler and model runner, binds the runner to the scheduler-owned KV cache, and advances the runtime one token-level step at a time. It does not own a tokenizer, does not decode text, and does not create a second KV cache.

## Main Files

- `include/tiny_llm/runtime/engine_core.h`
- `src/runtime/engine_core.cpp`
- `include/tiny_llm/runtime/processors.h`
- `include/tiny_llm/runtime/scheduler.h`

## Responsibilities

- Own the scheduler and model runner.
- Validate prompt token IDs against the model vocabulary exposed by `ModelRunner`.
- Convert `EngineCoreRequest` into scheduler `Request` state.
- Run one inference step by scheduling work, executing model forward/sampling, and updating scheduler state.
- Expose the most recent `RuntimeProfilingStats` produced by model execution.

Scheduling policy, request queues, preemption, and KV block allocation remain in `Scheduler`. Tensor preparation, model construction/forward, and sampling remain in `ModelRunner`.

## Attributes

- `scheduler_`: owned `Scheduler`, including request queues and KV cache manager.
- `runner_`: owned `ModelRunner`, bound to `scheduler_->kv_cache()` during construction.
- `last_step_profile_`: profiling data from the most recent `ModelRunner::run`. It is reset before no-op steps.

## Interfaces

- `EngineCore(const EngineArgs& args)`: constructs scheduler and model runner from the shared runtime construction object. This is the only construction path.
- `add_request(const EngineCoreRequest& request)`: validates prompt token range against model vocabulary and enqueues a scheduler request.
- `step()`: runs one scheduler/model/update cycle and returns `(outputs, has_scheduled_tokens)`.
- `last_step_profile()`: returns profiling for the last call to `step()`.

The constructor expects `EngineArgs` to contain either prebuilt runtime handles or model-construction fields understood by `Scheduler` and `ModelRunner`. Tokenizer pointers may be present in `EngineArgs` for adjacent layers, but `EngineCore` itself does not use tokenizer APIs.

## Step Semantics

`step()` performs:

1. Reset `last_step_profile_`.
2. Return an empty result and `false` if the scheduler has no unfinished requests.
3. Call `Scheduler::schedule()` to produce `SchedulerOutput`.
4. Record whether the scheduler selected any tokens this step.
5. Call `ModelRunner::run(scheduler_output)`.
6. Store `model_output.profiling`.
7. Move scheduler and model outputs into `Scheduler::update_from_output(...)`.
8. Convert ordered scheduler results into the unordered output map returned to callers.

The boolean return value is true when the scheduler had at least one scheduled token in this step. Callers can use it to distinguish a no-op step from chunked prefill work that has not emitted user-visible tokens yet.

## Request Conversion

`EngineCore::add_request()` builds a scheduler `Request` with:

- `request_id = EngineCoreRequest::internal_id`
- `prompt_token_ids = EngineCoreRequest::prompt_token_ids`
- `_all_token_ids = prompt_token_ids`
- `sampling_params = EngineCoreRequest::sampling_params`
- `status = WAITING`

The scheduler then resets generated-token state and starts token accounting from zero.
