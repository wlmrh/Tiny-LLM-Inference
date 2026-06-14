# LLMEngine Class

`LLMEngine` is the text-level runtime frontend over the token-only `EngineCore`. It accepts prompt strings and user sampling parameters, translates them into engine-core requests, and converts sampled token IDs back into user-facing text outputs.

## Project Position

`LLMEngine` sits between the high-level `LLM` facade and the token-level runtime core:

```text
caller / tools / tests
  -> LLM or direct LLMEngine usage
       -> LLMEngine
            -> InputPreprocessor
            -> EngineCore
                 -> Scheduler
                 -> ModelRunner
            -> OutPreprocessor
```

`LLM` owns deployment resources such as tokenizer, workspace, and raw KV memory. `LLMEngine` owns the frontend request-processing state and an `EngineCore`. The scheduler-owned `KVCache` and model execution state remain below `EngineCore`; `LLMEngine` does not create a second KV cache and does not inspect tensor-level scheduling metadata.

Direct `LLMEngine` construction is useful for tests, tools, and callers that already have an `EngineArgs` object. Deployment-style callers should usually use `LLM`, which wires model files, tokenizer, workspace, and KV memory before constructing `LLMEngine`.

## Responsibilities

- Validate frontend construction through `InputPreprocessor` and `OutPreprocessor`.
- Assign internal request IDs and bind optional external request IDs.
- Tokenize prompt strings and normalize `UserSamplingParams` into `SamplingParams`.
- Enqueue normalized `EngineCoreRequest` objects into `EngineCore`.
- Run one runtime step at a time.
- Incrementally decode sampled token IDs into `UserOutput` objects.
- Release external request ID bindings when frontend output state finishes.
- Expose profiling data from the most recent `EngineCore::step()`.

`LLMEngine` intentionally does not own tokenizer/model/workspace/KV resources passed through prebuilt `EngineArgs` pointers. It also does not make scheduling decisions; those belong to `Scheduler`, reached only through `EngineCore`.

## Public Interface

- `explicit LLMEngine(const EngineArgs& args)`: constructs `EngineCore`, `InputPreprocessor`, and `OutPreprocessor` from shared runtime arguments. The tokenizer must be available because both preprocessors depend on it.
- `~LLMEngine()`: destroys frontend state and the owned core. The destructor is out-of-line so the header can forward-declare `EngineCore`.
- Copy construction and copy assignment are disabled by owned runtime state. Move construction and move assignment are not exposed as public API.
- `add_request(const std::string& prompt, const UserSamplingParams& user_params, const std::string& ext_request_id)`: tokenizes and validates one prompt, assigns an internal ID, enqueues the core request, registers output state, and returns the internal ID. An empty external ID is replaced with a generated `req-<internal_id>` value.
- `has_unfinished_requests() const`: returns whether the frontend output processor still has unfinished request state. This is the condition used by `LLM` generation loops.
- `step()`: advances the token-level runtime once and returns zero or more `UserOutput` objects. A chunked prefill step can legitimately return no user output while work was scheduled.
- `last_step_profile() const`: returns the profile cached from the most recent user-visible `EngineCore::step()`.

## Private State

- `core_`: owned `EngineCore`, responsible for scheduler/model-runner orchestration over token IDs.
- `last_step_profile_`: cached profile copied from `EngineCore` immediately after the main step.
- `input_preprocessor_`: frontend request translator. It owns the internal request ID counter and the external-ID uniqueness map.
- `output_preprocessor_`: frontend output assembler. It owns per-request decoded text, generated token IDs, finish state, and stop-condition checks.

After the frontend output processor has no unfinished states, `LLMEngine::step()` performs one final `EngineCore::step()` cleanup call and verifies it produced no outputs or scheduled tokens. This keeps the public generation loop driven by frontend output state while still giving the core a final chance to release scheduler/KV state.

## Step Flow

`step()` performs:

1. Call `EngineCore::step()` to schedule and execute one token-level step.
2. Pass core outputs to `OutPreprocessor::process_outputs()`.
3. For every finished `UserOutput`, release its external request ID binding from `InputPreprocessor`.
4. If no frontend request remains unfinished, call one cleanup core step and assert that it is empty.
5. Return the user-facing outputs for this step.

The method is single-process and synchronous. It assumes callers repeatedly call `step()` while `has_unfinished_requests()` is true.

## Related Types

- `EngineArgs`: construction aggregate shared across `LLMEngine`, `EngineCore`, `Scheduler`, and `ModelRunner`.
- `UserSamplingParams`: frontend sampling settings. `max_tokens == 0` means use `EngineArgs::max_generated_tokens`.
- `EngineCoreRequest`: normalized token request consumed by `EngineCore`.
- `EngineCoreOutput`: sampled token output produced by `EngineCore`.
- `UserOutput`: decoded text output returned to callers.
