# Tiny-LLM-Inference Documentation

This directory contains the current English documentation for the Tiny-LLM-Inference runtime.

## Core Documents

- [Architecture](Architecture.md): end-to-end runtime architecture, ownership boundaries, rendered D2 architecture map, execution flow, data flow, and build layout.
- [Design Review Notes](Design_Review_Notes.md): implementation issues, redundant fields, and cleanup candidates found while documenting the current code.

## Module Documents

- [Runtime API](modules/Runtime_API.md): `LLM`, `LLMEngine`, request ingestion, streaming outputs, and offline usage.
- [LLM](modules/LLM.md): high-level offline facade positioning, construction options, public methods, outputs, and owned resources.
- [LLMEngine](modules/LLMEngine.md): text-level engine frontend positioning, request flow, public methods, and owned preprocessing state.
- [Engine Core](modules/Engine_Core.md): token-level orchestration between scheduling, model execution, and scheduler state updates.
- [Scheduler](modules/Scheduler.md): request queues, prefill/decode scheduling, KV block allocation, preemption, and outputs.
- [Request State](modules/Request_State.md): request lifecycle, token accounting, status transitions, and output state.
- [KV Cache and Memory](modules/KV_Cache_and_Memory.md): `KVCache`, `KVCacheManager`, `BlockAllocator`, and `StackAllocator`.
- [Model Runner](modules/Model_Runner.md): `SchedulerOutput` flattening, `PreparedInputs`, model invocation, sampling, profiling, and HF model construction.
- [Tokenizer and Processors](modules/Tokenizer_and_Processors.md): tokenizer contract, HuggingFace tokenizer loading, input validation, output decoding, and stop handling.
- [Sampling](modules/Sampling.md): default greedy sampling, repetition penalty, seeded non-greedy sampling, generation config, and CUDA/CPU sampling paths.
- [Models and Weights](modules/Models_and_Weights.md): LLaMA/Qwen-compatible model modules, HF config loading, safetensors loading, and weight binding.
- [Operators](modules/Operators.md): matmul, RMSNorm, LLaMA helper ops, RoPE, and paged attention backends.
- [Device and Execution Context](modules/Device_and_Execution_Context.md): CPU/CUDA device selection, execution context, runtime context, and profiling.
- [Tools, Tests, and Benchmarks](modules/Tools_Tests_and_Benchmarks.md): project executables, comparison scripts, CTest registration, and benchmark policy.

## Historical Documents

The older files in this directory are retained as historical implementation notes. Where a legacy file name overlaps a current module, it now points to the corresponding current module document.
