# Device and Execution Context Module

This module centralizes CPU/CUDA device selection and per-step execution resources.

## Main Files

- `include/tiny_llm/runtime/parallel_config.h`
- `src/runtime/parallel_config.cpp`
- `include/tiny_llm/core/context.h`
- `src/core/execution_context.cpp`
- `include/tiny_llm/runtime/runtime_context.h`
- `include/tiny_llm/runtime/runtime_stats.h`
- `include/tiny_llm/runtime/profiling.h`

## `ParallelConfig`

`ParallelConfig` is the runtime device configuration.

Attributes:

- `device_type_`: `kCPU` or `kCUDA`.
- `device_id_`: CPU must be `0`; CUDA must be non-negative.

Interfaces:

- `cpu()`
- `cuda(device_id)`
- `device_type()`, `device_id()`
- `is_cpu()`, `is_cuda()`
- `torch_device()`
- `validate()`
- equality operators.

`torch_device()` is the explicit conversion point to libtorch device types; most runtime configuration should stay on `device_type()` and `device_id()`.

The project currently supports one CPU device or one CUDA device per runtime.

## `ExecutionContext`

`ExecutionContext` is the low-level resource bundle used by operators.

Attributes:

- `stream_`: CUDA stream handle.
- `ws_`: non-owning `StackAllocator*`.
- `kv_`: non-owning `KVCache*`.
- `parallel_config_`: selected runtime device.
- `compute_dtype_`: compute dtype selected for this execution context.

Interfaces:

- `stream()`
- `workspace()`
- `kv()`
- `parallel_config()`
- `device()`
- `compute_dtype()`
- `begin_step()`
- `step_guard()`

`StepGuard` calls `begin_step()` on construction. This resets workspace allocations for the current model step.

## `RuntimeContext`

`RuntimeContext` is passed to `Model::forward`.

Attributes:

- `execution_`: `ExecutionContext&`.
- `attention_metadata_`: paged-attention tensor metadata for the current step.
- `profiling_stats_`: optional mutable profile sink.
- `profile_detail_enabled_`: enables per-component profiling only when stats are present.

Interfaces:

- `execution()`
- `device()`
- `attention_metadata()`
- `profiling_stats()`
- `profile_detail_enabled()`

This object is the only supported way to pass per-step attention metadata through model code. `ModelRunner` owns or binds an `ExecutionContext` explicitly; no process-wide execution-context compatibility layer is used.

## Profiling

`RuntimeProfilingStats` records:

- preparation, prefill, decode, sampling times;
- embedding, QKV projection, RoPE, attention, output projection, MLP, norm, and LM head times;
- prefill, decode, and sampled token counts.

`RuntimeProfilingStats::add()` aggregates stats across generation steps.

`ScopedRuntimeProfile` records model component timing when detailed profiling is enabled. The helper stores only the active stats pointer and device, while `RuntimeContext` remains the source of those values. CUDA paths synchronize around profiled regions to make timings meaningful.
