# Design Review Notes

These notes track design issues found during documentation and optimization work. Items listed as resolved have corresponding code and test updates; remaining items are architecture constraints to keep visible.

## Resolved Items

### `RequestData::block_ids`

Resolved: the redundant first-layer `block_ids` alias was removed. The scheduler-to-model contract now uses only all-layer `block_tables`.

### Scheduler Config Semantics

Resolved: waiting-request admission now honors `SchedulerConfig::max_running_requests`, and KV allocation failure only attempts tail preemption when `SchedulerConfig::enable_preemption` is true.

### Sampling Parameters

Resolved: `temperature`, `top_k`, and `top_p` are implemented for seeded sampling. Greedy behavior remains the default when `temperature == 0`.

### Sharded Safetensors Validation

Resolved: the `LLM` convenience facade accepts sharded `*.safetensors` directories when the default `model.safetensors` file is absent, matching `ModelRunner` discovery behavior.

### Frontend Request State

Resolved for completed requests: `OutPreprocessor` removes completed request state after final output delivery, and `InputPreprocessor` releases active external IDs after completion.

### KV Cache and Block Allocator Safety

Resolved: `KVCache::start_sequence()` rejects duplicate sequence IDs, and `BlockAllocator` tracks allocation state to reject invalid or duplicate frees.

### ModelRunner Public Surface and Step Scratch

Resolved: `ModelRunner` no longer exposes input preparation or raw vocabulary lookup. It keeps a narrow token-validation entry point for `EngineCore`, while per-step request IDs, sampling parameters, and token histories are carried by a local prepared batch instead of persistent runner members.

## Remaining Architecture Constraints

### Global Execution Context Limits Multi-Engine Safety

`ModelRunner` initializes and resets an internal process-wide execution context. This keeps operator compatibility but makes multiple simultaneous engine instances risky.

Suggested direction:

- Move all model/operator paths to explicit `ExecutionContext`/`RuntimeContext` passing.
- Keep the internal global execution context only as a compatibility fallback until removed.

### Legacy Thread-Local Paged Attention Metadata

`ops::set_paged_attention_runtime_metadata`, `clear_paged_attention_runtime_metadata`, `current_paged_attention_runtime_metadata`, and `PagedAttentionRuntimeMetadataGuard` still exist. The current model path passes metadata explicitly through `RuntimeContext`, and the legacy API is now marked as compatibility surface in the header.

Suggested cleanup:

- Remove the thread-local API after tests and helper tools use explicit metadata.

### Model Buffers Are Max-Batch Preallocated

`LlamaModel` preallocates reusable buffers sized by max batch/token count. This is simple and fast for the current single-process runtime, but the model throws if a flattened scheduled batch exceeds allocated capacity.

Suggested direction:

- Keep `ModelRunner::resolve_model_max_batch_size()` aligned with scheduler token budgets.
- Consider dynamic reallocation if future schedulers change batch sizing at runtime.
