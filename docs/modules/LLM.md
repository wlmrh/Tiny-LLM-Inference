# LLM Module

`LLM` is the top-level offline C++ facade for Tiny-LLM-Inference. It is the
entry point for deployment-style callers that have a HuggingFace model
directory and want to submit prompt strings without manually wiring tokenizer,
workspace, KV memory, scheduler, model runner, or model objects.

## Position in the Runtime

`LLM` sits above the text/token frontend and the token-level execution core:

```text
application code
  -> LLM
       -> LLMEngine
            -> InputPreprocessor
            -> EngineCore
                 -> Scheduler
                 -> ModelRunner
                      -> Model
            -> OutPreprocessor
```

The class owns deployment resources and passes non-owning handles into lower
layers through `EngineArgs`. The lower layers keep their specialized
responsibilities:

- `LLMEngine` owns text/token conversion, frontend request IDs, output assembly,
  and incremental decoding.
- `EngineCore` owns the token-level scheduler/model-runner step loop.
- `Scheduler` owns request state, waiting/running queues, and runtime KV cache
  state.
- `ModelRunner` prepares tensors, builds `RuntimeContext`, runs the model, and
  samples token IDs.

`LLM` is therefore a convenience boundary, not a second engine core. It should
not duplicate tokenization, scheduling, model execution, sampling, or per-request
KV-cache logic.

## Main Files

- `include/tiny_llm/runtime/llm.h`
- `src/runtime/llm.cpp`

## Responsibilities

- Normalize and validate construction options, including `~` expansion for the
  model path.
- Validate that the HuggingFace model directory has the files required by the
  HF runtime path.
- Load model config metadata needed to size the KV block pool.
- Own the tokenizer, per-step workspace allocator, raw CPU/CUDA KV memory pool,
  and `LLMEngine`.
- Translate single-prompt and batch `generate` calls into `LLMEngine` requests.
- Return final `CompletionOutput` values and, when requested, emit incremental
  callback events during the same `generate` call.
- Accumulate profiling counters for the most recent generation call.

## Public Types

### `LLMOptions`

`LLMOptions` configures construction of the complete offline runtime.

Attributes:

- `model`: HuggingFace model directory.
- `parallel_config`: CPU or selected CUDA device. CPU is the default.
- `weight_file`: requested safetensors file. The default is
  `model.safetensors`; the lower runtime can also use sorted safetensors shards
  for the default path.
- `max_num_seqs`: active sequence limit used when the scheduler config does not
  set `max_running_requests`.
- `max_tokens`: default generated-token limit when request sampling params leave
  `max_tokens` at `0`.
- `block_size_tokens`: token capacity per KV block.
- `kv_num_blocks`: number of physical KV blocks to allocate.
- `workspace_pool_size`: bytes reserved for the step-local `StackAllocator`.
- `scheduler_config`: lower-level scheduler overrides, including running request
  limit, preemption setting, and prefill token budget.

### Generation Types

- `LLMSamplingParams`: alias for `UserSamplingParams`. It keeps the public
  `LLM` API aligned with user-facing sampling settings. `max_tokens == 0` means
  "use `LLMOptions::max_tokens`."
- `CompletionOutput`: final per-prompt result. It contains the original prompt,
  cumulative decoded text, generated token IDs, finish flag, and finish reason.
- `CompletionStreamOutput`: incremental callback payload. It extends
  `CompletionOutput` with `prompt_index`, `delta_text`, and the latest
  `token_id`.
- `CompletionStreamCallback`: synchronous callback invoked for each user-facing
  incremental output event.

## Public Interface

Constructors:

- `LLM(std::string model)`: construct a CPU runtime from a model directory.
- `LLM(std::string model, ParallelConfig parallel_config)`: construct a runtime
  for CPU or a selected CUDA device.
- `LLM(LLMOptions options)`: construct with explicit resource, scheduler, and
  model-loading settings.

Generation methods:

- `generate(const std::vector<std::string>& prompts, const LLMSamplingParams& sampling_params = {}, CompletionStreamCallback callback = {})`:
  blocking batch generation.
- `generate(const std::string& prompt, const LLMSamplingParams& sampling_params = {}, CompletionStreamCallback callback = {})`:
  blocking single-prompt generation.

Profiling:

- `last_generation_profile()`: return profiling counters accumulated during the
  most recent generation call.

There is no separate public streaming-specific method. Streaming is an optional
behavior of `generate`: callers pass a `CompletionStreamCallback` when they need
incremental events, and omit the callback when they only need final outputs.
Batch calls use one `LLMSamplingParams` value for every prompt in the batch.

## Owned State and Lifecycle

`LLM` owns all resources needed by the high-level offline runtime.

Private attributes:

- `options_`: normalized construction options.
- `tokenizer_`: owned `HFLlamaTokenizer`, passed to `LLMEngine` through a
  non-owning pointer.
- `workspace_`: owned `StackAllocator` used by lower execution paths for
  per-step temporary tensors.
- `engine_`: owned `LLMEngine`.
- `kv_pool_`: raw CPU or CUDA memory backing scheduler-owned KV blocks.
- `last_generation_profile_`: accumulated runtime profile for the latest
  generation call.

Copy construction and copy assignment are disabled because duplicating an
initialized inference facade would make ownership of tokenizer state, engine
state, KV memory, and possibly CUDA memory ambiguous. Move construction and move
assignment transfer ownership of those resources and clear the moved-from raw KV
pool pointer to avoid double release.

Destruction and move assignment tear down `engine_` before workspace, tokenizer,
and KV memory so lower layers release their runtime state before backing
resources disappear.

## Construction Flow

Construction performs:

1. Expand `~` in `LLMOptions::model`.
2. Validate option values and model directory files.
3. Load `LlamaConfig` from the model directory.
4. Compute KV block size and total KV pool size.
5. Construct tokenizer, workspace allocator, and CPU/CUDA KV pool.
6. Fill `EngineArgs` with model-loading, device, scheduler, workspace, and KV
   fields.
7. Construct `LLMEngine`.

If any construction step fails after resource allocation begins, `LLM` releases
the partially constructed engine, workspace, tokenizer, and KV pool before
rethrowing the error.

## Generation Flow

`generate` performs:

1. Validate that the engine exists.
2. Return an empty result for an empty prompt batch.
3. Reset `last_generation_profile_`.
4. Add each prompt to `LLMEngine` and map the returned internal request ID to
   the original prompt index.
5. Repeatedly call `LLMEngine::step()` while frontend requests are unfinished.
6. Accumulate `LLMEngine::last_step_profile()` into
   `last_generation_profile_`.
7. Update the matching `CompletionOutput` for each returned `UserOutput`.
8. If a callback was provided, emit a `CompletionStreamOutput` for that event.
9. Return the final output vector in the same order as the input prompts.

The single-prompt overload wraps the prompt in a one-element vector and returns
the first final output.
