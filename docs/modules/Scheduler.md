# Scheduler Module

`Scheduler` owns request state between engine steps. It decides which token IDs are computed next, reserves KV cache slots for those tokens, and applies sampled model output back to requests.

## Main Files

- `include/tiny_llm/runtime/scheduler.h`
- `src/runtime/scheduler.cpp`
- `include/tiny_llm/runtime/request.h`
- `src/runtime/request.cpp`

## Responsibilities

- Maintain active `Request` objects and their `waiting`/`running` queues.
- Enforce the per-step scheduled-token budget.
- Split work into chunked prefill and one-token decode steps.
- Bind or own the runtime `KVCache` through `KVCacheManager`.
- Allocate KV blocks before a request is emitted in `SchedulerOutput`.
- Preempt tail running requests when KV capacity is insufficient and preemption is enabled.
- Consume `ModelRunnerOutput`, append sampled tokens, emit `EngineCoreOutput`, and release finished KV state.

The scheduler does not tokenize text, run model code, sample logits, or own a second KV cache for the same engine.

## Configuration

`SchedulerConfig` contains:

- `max_running_requests`: maximum active running request count; `0` means unlimited.
- `enable_preemption`: allows tail preemption on KV allocation failure.
- `max_prefill_tokens_per_step`: lower-bounded at `1` and used as the step token budget.

## KV Cache Manager

`KVCacheManager` is the scheduler's KV ownership adapter. It either binds the engine-provided cache or constructs one from `EngineArgs`; it is not a runner-local duplicate cache.

It provides block-count estimation, sequence start/end, slot allocation, and block-table refresh helpers. `refresh_block_tables(...)` is the active all-layer contract consumed by `ModelRunner`; `refresh_block_table(...)` remains as a legacy layer-0 helper.

## Scheduler Interface

- `Scheduler(SchedulerConfig)`: constructs scheduling policy state without binding KV cache.
- `Scheduler(const EngineArgs&)`: binds `args.kv` or constructs an owned KV cache from args.
- `Scheduler(KVCache*, SchedulerConfig)`: binds an existing cache.
- `add_request(Request)`: validates and enqueues a request.
- `schedule()`: creates one step-local `SchedulerOutput`.
- `update_from_output(SchedulerOutput, ModelRunnerOutput)`: advances request state after model execution.
- `get_num_unfinished_requests() const` and `has_unfinished_requests() const`: report active scheduler state.
- `kv_cache()`: exposes the scheduler-owned or scheduler-bound cache to `EngineCore`/`ModelRunner`.

## Step Output Contract

`SchedulerOutput` is the scheduler-to-runner package for one step:

- `scheduled_reqs`: ordered `RequestData` entries to execute.
- `num_scheduled_tokens`: request ID to scheduled token count.
- `total_num_scheduled_tokens`: flattened token count across all scheduled requests.
- `preempted_req_ids`: request IDs preempted while building this step.

`ModelRunner` consumes `scheduled_reqs`, `num_scheduled_tokens`, and `total_num_scheduled_tokens`. Scheduler/KV cleanup is handled by scheduler methods; the runner does not consume cleanup ID fields.

`RequestData` is step-local execution metadata:

- `req_id`: request ID.
- `new_token_ids`: prefill chunk tokens or the previous generated token for decode.
- `num_computed_tokens`: tokens with KV state before this step.
- `prompt_token_count`: original prompt length.
- `is_prefill`: true when the chunk still processes context tokens.
- `block_tables`: all-layer host table `[layer][logical_block] -> physical_block_id`.
- `sampling_params`: normalized sampling parameters for the request.
- `all_token_ids`: full prompt plus generated-token history before sampling this step.

## Scheduling Algorithm

`schedule()` first scans a snapshot of `running`, then scans `waiting` if no running request was preempted during the first phase.

For each eligible request:

- If `num_computed < all_token_count`, schedule a fair prefill chunk bounded by remaining step budget.
- Otherwise schedule one decode token by replaying the last token in `all_token_ids`.
- Reserve KV slots before the request is emitted.
- Build `RequestData` through the shared private helper so prefill and decode use the same metadata contract.

When a waiting request is admitted, it moves to running as part of KV slot allocation. `max_running_requests` limits only admission from waiting; existing running requests remain candidates.

## Preemption

If KV slot allocation fails and preemption is enabled, the scheduler chooses the tail running request as a victim. `preempt_request()` releases its KV sequence state, resets `num_computed` to `0`, removes it from `running`, and pushes it to the front of `waiting`.

The request's `all_token_ids` history is preserved, so recomputation includes the original prompt and already generated context.

## Updating From Model Output

`update_from_output()`:

1. Erases requests already marked finished.
2. Marks scheduled requests as running.
3. Advances prefill `num_computed` by the scheduled chunk size.
4. Reads sampled token IDs from `ModelRunnerOutput`, which is defined with the runner/runtime contract rather than in `scheduler.h`.
5. Appends sampled tokens to request history and emits `EngineCoreOutput`.
6. Advances decode `num_computed` after sampled-token append.
7. Finishes requests by stop token or max generated length.
8. Releases KV blocks and erases finished requests.

The final prefill row sample is emitted as the first generated token when a prompt completes.
