# Tiny-LLM-Inference Architecture

Tiny-LLM-Inference is a small, single-process inference engine inspired by vLLM. The public runtime accepts text prompts, converts them into token IDs, schedules prefill/decode work over active requests, runs a LLaMA-style causal language model, samples one token per scheduled sequence from normalized sampling parameters, and streams decoded text back to the caller.

The codebase is intentionally layered:

- `LLM` is the offline convenience facade. It owns tokenizer, workspace memory, KV block memory, and `LLMEngine`.
- `LLMEngine` is the text/token boundary. It handles prompt preprocessing, tokenizer validation, output decoding, and user-facing request IDs.
- `EngineCore` is the token-level runtime loop. It owns the `Scheduler` and `ModelRunner`.
- `Scheduler` owns request state, waiting/running queues, KV block allocation, chunked prefill/decode decisions, and completion cleanup.
- `ModelRunner` converts scheduler output into dense tensors, builds `RuntimeContext`, invokes `Model::forward`, and samples final rows. The default path is greedy; seeded non-greedy sampling supports temperature, top-k, and top-p.
- `Model` implementations are reusable `torch::nn::Module` graphs. The current production model path is `LlamaForCausalLM`, which also supports Qwen2-family checkpoint shapes used by Qwen2.5-1.5B-Instruct.
- `operators` provide CPU/CUDA tensor kernels and torch-backed fallback paths.

## Project Architecture Map

The main class relationships are maintained in [architecture.d2](architecture.d2) and rendered to [architecture.svg](architecture.svg). Regenerate the image after editing the source with `d2 docs/architecture.d2 docs/architecture.svg`.

![Tiny-LLM-Inference architecture](architecture.svg)

This diagram intentionally shows only the major classes and separates ownership from non-owning runtime references. Tiny-LLM-Inference is a single-process offline runtime: `LLMEngine` owns text/token I/O, `EngineCore` owns the scheduling/execution loop, `Scheduler` owns request state and owns or binds the runtime `KVCache`, and `ModelRunner` bridges scheduled token work into the `Model` interface. In the default HuggingFace path, that model is a constructed `LlamaForCausalLM`; in compatibility paths, `ModelRunner` may reference a prebuilt `Model`.

## Runtime Flow

The high-level call path is:

```text
LLM::generate
  -> LLMEngine::add_request
       -> InputPreprocessor::process_inputs
       -> EngineCore::add_request
       -> OutPreprocessor::add_request
  -> while LLMEngine::has_unfinished_requests()
       -> LLMEngine::step
            -> EngineCore::step
                 -> Scheduler::schedule
                 -> ModelRunner::run
                      -> private input preparation
                      -> Model::forward(PreparedInputs, RuntimeContext)
                      -> sample_greedy_rows
                 -> Scheduler::update_from_output
            -> OutPreprocessor::process_outputs
```

The engine uses one sampled row per scheduled request. During prefill, the final prefill row is sampled as the first generated token. During decode, the scheduler feeds the last generated token and samples the decode row.

## Ownership Boundaries

`LLM` owns deployment resources:

- `HFLlamaTokenizer`
- `StackAllocator` workspace
- raw KV memory pool
- `LLMEngine`

`LLMEngine` owns frontend processors and `EngineCore`.

`EngineCore` owns `Scheduler` and `ModelRunner`. The scheduler owns or binds the runtime `KVCache`; `ModelRunner` receives the scheduler-owned `KVCache*`. There should be only one KV cache backing an engine.

`ModelRunner` either binds an externally supplied `Model*` or constructs and owns `LlamaForCausalLM` from HuggingFace safetensors. It also owns safetensor loader objects when they are needed to keep memory-backed tensor views alive.

## Data Model

The runtime separates request state from model inputs:

- `Request` is long-lived scheduler state.
- `RequestData` is one-step scheduler output for a request.
- `SchedulerOutput` is a full step batch.
- `PreparedInputs` is the tensor package consumed by model code.
- `RuntimeContext` carries execution/device handles, KV cache access, attention metadata, and optional profiling state.
- `ModelRunnerOutput` returns sampled token IDs and profiling data.
- `EngineCoreOutput` returns per-request token results to the frontend.
- `UserOutput` returns decoded text deltas and finish metadata to users.

The important tensor package produced inside `ModelRunner::run` is:

```text
input_ids     [num_total_tokens]                         int32
positions     [num_total_tokens]                         int32
slot_mapping  [num_total_tokens]                         int32
seq_indices   [num_total_tokens]                         int32
context_lens  [num_seqs]                                 int32
block_tables  [num_layers, num_seqs, max_blocks_per_seq] int32
```

These tensors are token-aligned except for `context_lens` and `block_tables`.

`PreparedInputs` may also carry derived full-prefill segment descriptors for the current step. These descriptors summarize contiguous full-prefill rows and are valid only for full prefill from position zero. They let CUDA full-prefill attention select the SDPA path without rereading CUDA metadata on the CPU in every decoder layer; they do not change the scheduler output contract.

## Scheduling Model

The scheduler currently implements FCFS-style scheduling over two queues:

- `waiting`: new or preempted requests.
- `running`: requests with active KV state.

Each `schedule()` call consumes a per-step token budget. Running requests are considered first. If no running request was preempted while scheduling the running queue, waiting requests are then admitted.

Request work is split into:

- prefill chunks: one or more prompt/context tokens, limited by `max_prefill_tokens_per_step`;
- decode steps: one generated-token replay per running request.

KV capacity is allocated before a request is added to the step output. If there is not enough KV capacity, the scheduler preempts a tail running request, releases its KV state, resets its `num_computed`, and pushes it back to the front of `waiting` so its full prompt plus generated context can be recomputed.

## KV Cache Model

`KVCache` manages metadata for paged KV storage. Physical block memory is owned by `BlockAllocator`; `KVCache` stores per-sequence, per-layer page tables that map logical blocks to physical block IDs.

Each physical block stores both key and value data:

```text
2 * block_size_tokens * (num_key_value_heads * head_dim) * sizeof(float)
```

The attention runtime receives `slot_mapping`, `seq_indices`, `context_lens`, rank-3 `block_tables`, and optional derived prefill segments through `RuntimeContext::attention_metadata()`. This is the current preferred path. Legacy thread-local paged-attention metadata APIs still exist for compatibility.

## Model Runtime

`LlamaForCausalLM` is the current causal LM implementation. It owns `LlamaModel` and an LM head `Linear`. `LlamaModel` contains token embedding, the vector of `LlamaDecoderLayer`, and final `RMSNorm`.

Each decoder layer contains:

- input `RMSNorm`
- self-attention with stacked Q/K/V projection, RoPE, paged attention, and output projection
- post-attention `RMSNorm`
- MLP with stacked gate/up projection, SiLU multiply, and down projection

The model path supports LLaMA-family and Qwen2-family checkpoints that match this layout, including GQA and optional Q/K/V projection bias.

## Device Model

`ParallelConfig` selects CPU or one CUDA device. Tensor and allocator paths dispatch by actual tensor/device state, not compile flag alone. CPU tensors remain valid in CUDA builds.

CUDA builds add CUDA kernels for selected operators while retaining torch or CPU implementations where appropriate. The paged attention CUDA path now includes custom paged-attention kernels and a full-prefill SDPA path when precomputed segment descriptors are valid; fallback/reference paths remain for unsupported cases. This is still an evolving implementation, not a final optimized kernel.

## Build Layout

The build produces one static library target:

- `tiny_llm`

Compatibility aliases are kept for older tests or callers:

- `tiny_llm_core`
- `tiny_llm_models`
- `tiny_llm_operators`

The top-level library includes runtime, model, operator, and core allocator/context sources. `tests/CMakeLists.txt` fetches GoogleTest and registers unit, integration, smoke, and Transformers comparison tests. `benchmark/` contains benchmark binaries and Python wrappers but benchmark runs are not regular CTest tests.

## Supported Runtime Scope

Current supported behavior:

- single-process C++ runtime;
- CPU runtime by default;
- optional single-device CUDA runtime;
- HuggingFace `tokenizer.json` or `tokenizer.model`;
- LLaMA/SmolLM2-compatible checkpoints;
- Qwen2-family checkpoints with tied embeddings, GQA, projection bias, large RoPE theta, and optional `unk_token_id`;
- single-file `model.safetensors` and sorted sharded `*.safetensors` in `ModelRunner`;
- default greedy decoding, HuggingFace-style repetition penalty, seeded temperature/top-k/top-p sampling, and stop-token/max-length termination.

Important current limitations are recorded in [Design Review Notes](Design_Review_Notes.md).
