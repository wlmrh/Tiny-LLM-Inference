# LLM Class

`LLM` is the high-level offline facade for deployment-style C++ usage. It is the entry point callers use when they want to provide prompt strings and receive decoded completions without manually constructing tokenizer, scheduler, model runner, execution context, or KV-cache components.

## Project Position

`LLM` sits above `LLMEngine` in the runtime stack:

```text
caller
  -> LLM
       -> LLMEngine
            -> EngineCore
                 -> Scheduler
                 -> ModelRunner
                      -> Model
```

This class owns deployment resources and passes non-owning handles into lower layers through `EngineArgs`. Lower layers retain the token-level responsibilities: `LLMEngine` handles text/token conversion, `EngineCore` drives scheduling and execution, `Scheduler` owns request/KV state, and `ModelRunner` prepares tensors and invokes the model.

## Responsibilities

- Normalize construction options, including `~` expansion for the model path.
- Validate model directory inputs before runtime construction.
- Load HuggingFace LLaMA/Qwen-compatible config data needed for KV sizing.
- Allocate tokenizer, per-step workspace, raw CPU/CUDA KV block memory, and `LLMEngine`.
- Translate batch and single-prompt generation calls into engine requests.
- Produce full `CompletionOutput` results or invoke `CompletionStreamCallback` for incremental events.
- Aggregate `RuntimeProfilingStats` across the latest generation call.

`LLM` intentionally does not expose scheduler queues, request state, tensor preparation, or model-forward APIs. Those remain owned by lower runtime modules.

## Construction Options

`LLMOptions` configures the facade:

- `model`: HuggingFace model directory. The constructor expands leading `~`.
- `parallel_config`: CPU or single CUDA device selection.
- `weight_file`: requested safetensors file. The default `model.safetensors` path also allows the lower model runner to fall back to sorted safetensors shards.
- `max_num_seqs`: default active sequence limit used when scheduler config does not set one.
- `max_tokens`: default generated-token limit when user sampling params do not override it.
- `block_size_tokens`: token capacity per KV block.
- `kv_num_blocks`: number of physical KV blocks to allocate.
- `workspace_pool_size`: bytes reserved for the step-local `StackAllocator`.
- `scheduler_config`: low-level scheduler overrides for policy, preemption, running-request limit, and prefill token budget through `max_prefill_tokens_per_step`.

The top-level sequence and generated-token limits are convenience defaults. If `scheduler_config.max_running_requests` or `scheduler_config.max_prefill_tokens_per_step` is explicitly set, that lower-level value is preserved.

## Output Types

- `LLMSamplingParams`: public alias for `UserSamplingParams`, used by `LLM` methods so callers do not need to think in processor-layer naming. Leave `max_tokens` at `0` to use `LLMOptions.max_tokens`; set a positive value to override it for one generation call.
- `CompletionOutput`: final per-prompt result containing the original prompt, full decoded text, generated token IDs, finish flag, and finish reason.
- `CompletionStreamOutput`: incremental callback payload that extends `CompletionOutput` with prompt index, decoded delta, and latest token ID.
- `CompletionStreamCallback`: `std::function<void(const CompletionStreamOutput&)>` invoked for each user-facing engine output event when streaming is requested.

Prefer default construction plus field assignment for sampling params. Positional aggregate initialization is not part of the stable public style because the user-facing and normalized sampling structs share fields through a public base struct.

## Public Interface

- `LLM(std::string model)`: construct a CPU runtime from a model directory.
- `LLM(std::string model, ParallelConfig parallel_config)`: construct a runtime for CPU or a selected CUDA device.
- `LLM(LLMOptions options)`: construct with explicit resource, scheduler, and model-loading settings.
- `generate(const std::vector<std::string>& prompts, const LLMSamplingParams& sampling_params)`: blocking batch generation. It delegates to `generate_stream` with an empty callback.
- `generate(const std::string& prompt, const LLMSamplingParams& sampling_params)`: blocking single-prompt generation.
- `generate_stream(const std::vector<std::string>& prompts, const LLMSamplingParams& sampling_params, CompletionStreamCallback callback)`: batch generation with incremental callback events.
- `generate_stream(const std::string& prompt, const LLMSamplingParams& sampling_params, CompletionStreamCallback callback)`: single-prompt streaming wrapper.
- `last_generation_profile()`: returns profiling counters accumulated during the most recent generation call.

`LLM` is move-only. Copy construction and copy assignment are disabled because the object owns raw CPU/CUDA KV memory and lower-layer runtime state.

## Private State

- `options_`: normalized construction options.
- `tokenizer_`: owned `HFLlamaTokenizer` used by `LLMEngine` processors through non-owning pointers.
- `workspace_`: owned `StackAllocator` for per-step temporary tensors.
- `engine_`: owned `LLMEngine`.
- `kv_pool_`: raw CPU or CUDA memory pool backing scheduler-owned KV blocks.
- `last_generation_profile_`: accumulated runtime profile for the latest generation call.

Destruction and move assignment tear down `engine_` before workspace/tokenizer/KV memory so lower layers release scheduler and model-runner state before their backing resources disappear.

`LLM` maps outputs back to prompt indices with the internal request IDs returned by `LLMEngine::add_request()`. It does not maintain a second request-id counter at the facade layer.
