# LLM Class

`LLM` is the high-level offline facade for deployment-style C++ usage. It is
the entry point callers use when they have a HuggingFace model directory and
want to pass prompt strings in and receive decoded completions back, without
manually constructing tokenizer, workspace, scheduler, model runner, execution
context, or KV-cache components.

## Project Position

`LLM` sits at the top of the runtime stack:

```text
caller
  -> LLM
       -> LLMEngine
            -> EngineCore
                 -> Scheduler
                 -> ModelRunner
                      -> Model
```

The class owns deployment resources and passes non-owning handles into lower
layers through `EngineArgs`. Lower layers retain the token-level
responsibilities: `LLMEngine` handles text/token conversion, `EngineCore` drives
scheduling and execution, `Scheduler` owns request and KV-cache state, and
`ModelRunner` prepares tensors, invokes the model, and samples tokens.

Main files:

- `include/tiny_llm/runtime/llm.h`
- `src/runtime/llm.cpp`

## Responsibilities

- Validate and normalize construction options, including model path handling.
- Check that the model directory has the files needed by the HF runtime path.
- Load enough model config metadata to size the KV-cache backing pool.
- Own the tokenizer, per-step workspace, raw CPU/CUDA KV memory pool, and
  `LLMEngine`.
- Translate single-prompt and batch generation calls into `LLMEngine` requests.
- Return final `CompletionOutput` values or invoke a streaming callback for
  incremental output events.
- Accumulate profiling counters for the most recent generation call.

`LLM` does not implement tokenization, scheduling, model execution, sampling, or
per-request KV-cache state. Those responsibilities remain in `LLMEngine`,
`EngineCore`, `Scheduler`, and `ModelRunner`.

## Public Types

### `LLMOptions`

`LLMOptions` configures the convenience facade.

- `model`: HuggingFace model directory.
- `parallel_config`: CPU or selected CUDA device. CPU is the default.
- `weight_file`: requested safetensors file. The default is
  `model.safetensors`; the lower runtime can also use sorted safetensors shards
  for the default path.
- `max_num_seqs`: default active sequence limit used when the scheduler config
  does not set one.
- `max_tokens`: default generated-token limit when user sampling params leave
  `max_tokens` at `0`.
- `block_size_tokens`: token capacity per KV block.
- `kv_num_blocks`: number of physical KV blocks to allocate.
- `workspace_pool_size`: bytes reserved for the step-local `StackAllocator`.
- `scheduler_config`: lower-level scheduler overrides such as running request
  limit, preemption setting, and prefill token budget.

### Generation Types

- `LLMSamplingParams`: alias for `UserSamplingParams`. It keeps the public
  `LLM` API independent from processor naming while using the same sampling
  fields. `max_tokens == 0` means "use `LLMOptions::max_tokens`."
- `CompletionOutput`: final per-prompt result containing the original prompt,
  cumulative decoded text, generated token IDs, finish flag, and finish reason.
- `CompletionStreamOutput`: callback payload for incremental generation. It
  extends `CompletionOutput` with `prompt_index`, `delta_text`, and the latest
  `token_id`.
- `CompletionStreamCallback`: callback invoked synchronously for user-facing
  streaming events.

## Public Interface

- `LLM(std::string model)`: construct a CPU runtime from a model directory.
- `LLM(std::string model, ParallelConfig parallel_config)`: construct a runtime
  for CPU or a selected CUDA device.
- `LLM(LLMOptions options)`: construct with explicit resource, scheduler, and
  model-loading settings.
- `generate(const std::vector<std::string>& prompts, const LLMSamplingParams&)`:
  blocking batch generation.
- `generate(const std::string& prompt, const LLMSamplingParams&)`: blocking
  single-prompt generation.
- `generate_stream(...)` for a prompt batch: generation with incremental
  callback events.
- `generate_stream(...)` for a single prompt: single-prompt streaming wrapper.
- `last_generation_profile()`: returns profiling counters accumulated during
  the most recent generation call.

The blocking `generate` methods delegate to the streaming path with no
callback. Batch calls use one `LLMSamplingParams` value for every prompt in the
batch.

## Owned State and Lifecycle

`LLM` is move-only. Copy construction and copy assignment are disabled because
the object owns raw CPU/CUDA memory and lower-layer runtime state.

Private state:

- `options_`: normalized construction options.
- `tokenizer_`: owned `HFLlamaTokenizer`, passed to `LLMEngine` through a
  non-owning pointer.
- `workspace_`: owned `StackAllocator` used by lower execution paths for
  per-step temporary tensors.
- `engine_`: owned `LLMEngine`.
- `kv_pool_`: raw CPU or CUDA memory backing scheduler-owned KV blocks.
- `last_generation_profile_`: accumulated runtime profile for the latest
  generation call.

Destruction and move assignment tear down `engine_` before workspace,
tokenizer, and KV memory so lower layers release their runtime state before the
backing resources disappear.

## Generation Flow

`generate_stream` adds each prompt to `LLMEngine`, records the internal request
ID returned for each prompt index, and repeatedly calls `LLMEngine::step()` while
frontend requests remain unfinished. Each returned `UserOutput` updates the
matching `CompletionOutput`; when a callback is present, `LLM` emits a
`CompletionStreamOutput` for that user-visible event. The final result vector
preserves the input prompt order.
