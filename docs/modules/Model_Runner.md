# Model Runner Module

`ModelRunner` bridges scheduler output and model execution. It transforms request-oriented scheduler data into tensor-oriented model inputs, invokes the model, and samples output logits.

## Main Files

- `include/tiny_llm/runtime/model_runner.h`
- `src/runtime/model_runner.cpp`
- `include/tiny_llm/runtime/prepared_inputs.h`
- `include/tiny_llm/runtime/runtime_context.h`

## Responsibilities

- Construct or bind the runtime model.
- Keep safetensor loaders alive when zero-copy or memory-backed tensor views require them.
- Convert `SchedulerOutput` into `PreparedInputs`.
- Build paged-attention runtime metadata.
- Reset per-step workspace through `ExecutionContext::StepGuard`.
- Run `Model::forward`.
- Greedily sample selected rows with optional repetition penalty.
- Return sampled token IDs and profiling data.

## Attributes

- `owned_model_`: model constructed from `EngineArgs` when no prebuilt model is supplied.
- `owned_hf_loader_`: legacy single-loader ownership slot.
- `owned_hf_loaders_`: loaders for one or more safetensor shards.
- `model_`: active model pointer.
- `kv_`: scheduler-owned KV cache pointer.
- `kv_block_size_tokens_`: token capacity per KV block.
- `prepared_req_ids_`: request IDs corresponding to sample rows.
- `prepared_sampling_params_`: per-sample sampling parameters.
- `prepared_token_histories_`: per-sample histories used for repetition penalty.
- `debug_step_index_`: index used for optional debug log output.

## Interfaces

- `ModelRunner(const EngineArgs&, KVCache*)`: runtime constructor used by `EngineCore`.
- `ModelRunner(Model*, ExecutionContext*, KVCache*)`: test/prebuilt constructor.
- `vocab_size()`: returns active model vocabulary size.
- `prepare_inputs(const SchedulerOutput&)`: builds `PreparedInputs`.
- `run(const SchedulerOutput&)`: prepares inputs, runs model, samples logits, and returns `ModelRunnerOutput`.

## HF Model Construction

When `args.model == nullptr` and `args.model_type == EngineModelType::kHFLlamaSafeTensor`, `ModelRunner`:

1. Loads `LlamaConfig` from the HF directory.
2. Resolves `model.safetensors` or sorted `*.safetensors` shards.
3. Loads tensors into a `WeightMap` on the target device.
4. Constructs `LlamaForCausalLM`.
5. Allocates model buffers for the resolved maximum batch size.

The max model batch size is at least `args.max_batch_size` and at least `scheduler_config.max_prefill_tokens_per_step`.

## `PreparedInputs`

`PreparedInputs` is model-facing input state.

Attributes:

- `input_ids`: flattened token IDs, shape `[num_total_tokens]`, int32.
- `positions`: position of each token within its request context, shape `[num_total_tokens]`, int32.
- `slot_mapping`: physical KV slot for each token, shape `[num_total_tokens]`, int32.
- `seq_indices`: dense sequence index for each token in the step batch, shape `[num_total_tokens]`, int32.
- `context_lens`: context length per scheduled sequence, shape `[num_seqs]`, int32.
- `block_tables`: per-layer page tables, shape `[num_layers, num_seqs, max_blocks_per_seq]`, int32.
- `sample_row_offsets`: final row offset for each scheduled request.
- `prefill_segments`: optional derived descriptors `{row_start, seq_index, length}` for contiguous full-prefill rows.
- `prefill_segments_valid`: true only when the current step is entirely full prefill from position zero.

For an empty scheduler step, `PreparedInputs` contains defined zero-length tensors so downstream validation can stay uniform.

## Flattening Rules

For each scheduled request, `prepare_inputs()`:

- validates scheduled token counts and block table layer count;
- sets `context_len = num_computed_tokens + scheduled_tokens`;
- copies all layer block tables into a padded rank-3 tensor;
- writes one row per scheduled token;
- computes `position = num_computed_tokens + token_index`;
- computes `logical_block_index = position / kv_block_size_tokens`;
- computes `slot = physical_block_id * kv_block_size_tokens + (position % kv_block_size_tokens)`;
- appends the final token row to `sample_row_offsets`;
- precomputes full-prefill segments only when all scheduled requests are full prefill chunks with `num_computed_tokens == 0` and the flattened token count is large enough for the CUDA SDPA prefill path.

`sample_row_offsets` is the reason prefill can emit the first generated token from the last prefill row.

## Runtime Context and Model Invocation

`run_model()` builds `ops::PagedAttentionRuntimeMetadata` from the prepared tensors and then creates:

```cpp
RuntimeContext runtime_ctx(exec_ctx, metadata, profiling, detail_profile_enabled);
```

The model receives `PreparedInputs` plus this context. Attention modules read paged metadata from `RuntimeContext::attention_metadata()`.

The metadata includes tensor pointers for paged attention and, when valid, a pointer/count view over `PreparedInputs::prefill_segments`. The segment view is step-local and must not outlive `PreparedInputs`.

## Sampling and Output

`run()` calls:

```cpp
sample_greedy_rows(logits,
                   inputs.sample_row_offsets,
                   model_->vocab_size(),
                   &prepared_token_histories_,
                   &prepared_sampling_params_);
```

It then maps sampled token IDs back to request IDs through `req_id_to_index`.

## Profiling

`ModelRunner` records:

- input preparation time;
- model time split proportionally between prefill and decode tokens;
- sampling time;
- token counts;
- optional per-component model timings when `TINYLLM_PROFILE_DETAIL` is enabled.

CUDA profiling paths synchronize the selected device around measured regions.
