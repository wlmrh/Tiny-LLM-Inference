# Scheduler Module

The scheduler owns request state and decides which tokens are computed in each engine step. It also coordinates KV cache block allocation and preemption.

## Main Files

- `include/tiny_llm/runtime/scheduler_config.h`
- `include/tiny_llm/runtime/scheduler.h`
- `src/runtime/scheduler.cpp`
- `include/tiny_llm/runtime/request.h`

## Responsibilities

- Maintain all active `Request` objects.
- Maintain `waiting` and `running` queues.
- Enforce the per-step scheduled-token budget.
- Split request work into chunked prefill and decode.
- Allocate KV cache blocks before model execution.
- Preempt tail running requests when KV capacity is insufficient.
- Apply scheduler state updates after sampled tokens return from `ModelRunner`.
- Release finished request KV cache state.

## `SchedulerConfig`

`SchedulerConfig` lives in `scheduler_config.h` so construction APIs can accept scheduler settings without pulling in the full scheduler state/output interface.

Attributes:

- `max_running_requests`: maximum active running request count; `0` means unlimited.
- `enable_preemption`: allows tail preemption when KV allocation fails.
- `max_prefill_tokens_per_step`: token budget used by `Scheduler` as `max_num_scheduled_tokens`.

## `Scheduler`

Important attributes:

- `kvcache_manager`: binds or owns the runtime `KVCache`.
- `requests`: map from request ID to `Request`.
- `waiting`: queue of new or preempted request IDs.
- `running`: queue of request IDs with active runtime state.
- `max_num_scheduled_tokens`: per-step token budget.

Main interfaces:

- `Scheduler(const EngineArgs&)`: binds `args.kv` or constructs an owned KV cache from args.
- `Scheduler(KVCache*, SchedulerConfig)`: binds an existing cache.
- `add_request(Request request)`: validates and enqueues a request.
- `schedule()`: produces one `SchedulerOutput`.
- `update_from_output(SchedulerOutput, ModelRunnerOutput)`: consumes sampled tokens and returns generated `EngineCoreOutput` messages.
- `has_unfinished_requests() const`: true when any request is waiting or running.
- `kv_cache()`: exposes the bound cache for `ModelRunner`.

## `SchedulerOutput`

Attributes:

- `scheduled_reqs`: list of `RequestData` entries to execute this step.
- `num_scheduled_tokens`: request ID to token count.
- `total_num_scheduled_tokens`: total flattened token count.

Scheduler/KV cleanup happens inside scheduler methods; no cleanup IDs are exported to `ModelRunner`.

## `RequestData`

Attributes:

- `req_id`: request ID.
- `new_token_ids`: tokens to run in this step.
- `num_computed_tokens`: number of tokens whose KV state already exists before this step.
- `prompt_token_count`: original prompt length.
- `block_tables`: rank-2 host table `[layer][logical_block] -> physical_block_id`.
- `sampling_params`: normalized request sampling parameters.
- `context_token_ids`: full prompt plus generated-token history before sampling this step.

`ModelRunner` uses the all-layer `block_tables` contract directly.

## Scheduling Algorithm

`schedule()` runs in two phases:

1. Iterate a snapshot of `running`.
2. If no preemption occurred while scheduling running requests, iterate a snapshot of `waiting`.

For each request:

- If `num_computed_tokens < context_token_count`, schedule a prefill chunk.
- Otherwise schedule one decode token by replaying the last token in `context_token_ids`.
- Allocate enough KV slots before adding the request to the output.
- Refresh all per-layer block tables after allocation.

The prefill chunk size is bounded by the remaining token budget. Decode always schedules one token per selected request.

## Preemption

When `KVCacheManager::allocate_slots()` fails, the scheduler scans the back of `running` for a tail victim. The victim is reset as follows:

- KV sequence state is ended and blocks are released.
- `status` becomes `WAITING`.
- `num_computed_tokens` becomes `0`.
- the request is removed from `running` and pushed to the front of `waiting`.

The request's `context_token_ids` is preserved. This means recomputation includes the original prompt and already-generated context, which avoids losing generated tokens after preemption.

## Updating From Model Output

`update_from_output()` performs:

1. Cleanup of requests that were already `FINISHED`.
2. Mark scheduled requests as running.
3. Advance `num_computed_tokens` for prefill tokens.
4. Read sampled token IDs from `ModelRunnerOutput`.
5. Append sampled token to `context_token_ids`.
6. Emit `EngineCoreOutput`.
7. Finish requests by stop token or max generated length.
8. Release KV blocks and erase finished requests.

During prefill, the final prefill row sample is emitted as the first generated token. During decode, `num_computed_tokens` advances after appending the sampled token.
