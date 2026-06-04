# Runtime API Module

The runtime API module is the user-facing layer for offline generation. It hides tokenizer/model/KV wiring and exposes a small vLLM-style C++ interface.

## Main Files

- `include/tiny_llm/runtime/llm.h`
- `src/runtime/llm.cpp`
- `include/tiny_llm/runtime/engine.h`
- `src/runtime/engine.cpp`
- `include/tiny_llm/runtime/engine_args.h`

Class-level details for the high-level facade are maintained in [LLM](LLM.md), and the text-level engine frontend is documented in [LLMEngine](LLMEngine.md).

## Responsibilities

- Construct a complete runtime from a HuggingFace model directory.
- Own deployment resources such as tokenizer, workspace allocator, KV memory pool, and engine object.
- Accept string prompts and user sampling parameters.
- Return full completion outputs or invoke a per-step streaming callback.
- Aggregate profiling statistics across a generation call.

## `LLMOptions`

`LLMOptions` configures the convenience facade.

Key attributes:

- `model`: HuggingFace model directory.
- `parallel_config`: CPU or CUDA device configuration.
- `weight_file`: requested safetensors weight file, defaulting to `model.safetensors`.
- `max_num_seqs`: maximum logical active sequence count used to fill scheduler defaults.
- `max_num_batched_tokens`: default prefill token budget per scheduler step.
- `max_tokens`: default maximum generated tokens when user params do not override it.
- `block_size_tokens`: tokens per KV cache block.
- `kv_num_blocks`: physical KV block count.
- `workspace_pool_size`: bytes reserved for per-step workspace.
- `scheduler_config`: low-level scheduler options.

## `LLM`

`LLM` is a move-only offline facade.

Important attributes:

- `options_`: normalized construction options.
- `tokenizer_`: owned `HFLlamaTokenizer`.
- `workspace_`: owned `StackAllocator`.
- `engine_`: owned `LLMEngine`.
- `kv_pool_`: raw CPU or CUDA KV memory pool.
- `last_generation_profile_`: accumulated runtime profile for the most recent generation call.

Main interfaces:

- `LLM(std::string model)`: constructs a CPU runtime from a model path.
- `LLM(std::string model, ParallelConfig parallel_config)`: constructs a runtime for the selected device.
- `LLM(LLMOptions options)`: constructs a runtime with explicit resource settings.
- `generate(const std::vector<std::string>&, const LLMSamplingParams&)`: blocking batch generation.
- `generate(const std::string&, const LLMSamplingParams&)`: blocking single-prompt generation.
- `generate_stream(...)`: generation with `CompletionStreamCallback` invoked for each emitted token event.
- `last_generation_profile()`: returns accumulated timings and token counters.

`LLMSamplingParams` is an alias for `UserSamplingParams`. A `max_tokens` value of `0` uses the runtime default from `LLMOptions.max_tokens`; positive values override it for the request.

Construction behavior:

1. Expands `~` in the model path.
2. Validates option values.
3. Loads model config.
4. Computes KV block size and total KV pool size.
5. Constructs tokenizer, workspace, KV pool, and `LLMEngine`.
6. Fills `EngineArgs` with HF model construction fields and scheduler configuration.

## `LLMEngine`

`LLMEngine` is the string/token bridge over `EngineCore`. Class-level details are maintained in [LLMEngine](LLMEngine.md).

Important attributes:

- `core_`: owned `EngineCore`.
- `input_preprocessor_`: prompt tokenization, request ID assignment, external-ID tracking, and sampling normalization.
- `output_preprocessor_`: incremental decoding, output state, and finish detection.

Main interfaces:

- `add_request(prompt, user_params, ext_request_id)`: tokenizes and validates a prompt, adds it to core runtime, and registers output state.
- `has_unfinished_requests()`: checks frontend output state.
- `step()`: advances the runtime once and returns user-facing outputs.
- `last_step_profile()`: returns the most recent profile from `EngineCore`.

Finished scheduler/KV state is released during `Scheduler::update_from_output()`. `LLMEngine::step()` does not run a hidden cleanup step after final user output.

## `EngineArgs`

`EngineArgs` is the low-level construction object shared by `LLMEngine`, `EngineCore`, `Scheduler`, and `ModelRunner`.

Important fields:

- Prebuilt handles: `model`, `ctx`, `kv`, `tokenizer`.
- Device: `parallel_config`.
- Model construction: `model_type`, `hf_model_dir`, `hf_weight_file`, `max_batch_size`.
- Execution construction: `execution_stream`, `workspace`, `workspace_pool_size`.
- KV construction: `kv_num_layers`, `kv_block_size_tokens`, `kv_num_blocks`, `kv_block_size_bytes`, `kv_memory_pool`.
- Generation/scheduling: `max_generated_tokens`, `scheduler_config`.

If `model` is null and `model_type` is `kHFLlamaSafeTensor`, `ModelRunner` constructs the model from the HF directory. If `ctx` or `kv` is null, the runtime path constructs/binds the missing pieces from the resource fields.
