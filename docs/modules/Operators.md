# Operators Module

The operators module implements model math and attention kernels used by reusable model layers.

## Main Files

- `include/tiny_llm/operators/matmul.h`
- `src/operators/matmul/matmul.cpp`
- `src/operators/matmul/gemm_kernels.cu`
- `include/tiny_llm/operators/rmsnorm.h`
- `src/operators/rmsnorm/rmsnorm.cpp`
- `src/operators/rmsnorm/rmsnorm.cu`
- `include/tiny_llm/operators/llama_ops.h`
- `src/operators/llama_ops.cpp`
- `src/operators/llama_ops_kernels.cu`
- `include/tiny_llm/operators/paged_attention.h`
- `src/operators/paged_attention/*.cpp`
- `src/operators/paged_attention/*.cu`

## Matmul

Interface:

- `ops::gemm(const Tensor& a, const Tensor& b, Tensor& c, ExecutionContext& ctx)`

Responsibilities:

- validate tensor shapes and dtype through callers;
- dispatch to CPU/torch/CUDA implementations depending on tensor device and build flags;
- support `Linear` modules and LM head projection.

## RMSNorm

Interface:

- `ops::rmsnorm(const Tensor& x, const Tensor& w, Tensor& y, ExecutionContext& ctx, float eps)`

Responsibilities:

- compute row-wise RMS normalization;
- use CUDA kernel when available and tensors are CUDA;
- keep CPU path valid in CUDA builds.

## LLaMA Helper Ops

Interfaces:

- `embedding_lookup(ids, embedding, out, vocab_size, hidden_size, embedding_is_vocab_hidden)`
- `split_qkv(qkv, q, k, v, hidden_size, kv_hidden_size)`
- `apply_rope(...)`
- `silu_multiply(gate, up, out)`
- `copy_tensor(src, dst)`
- `add_tensors(lhs, rhs, out)`

Responsibilities:

- provide small validated tensor transformations used inside model modules;
- dispatch CUDA kernels for selected fast paths such as cached RoPE and SiLU multiply;
- use torch operations as device-aware fallback for some CUDA cases;
- preserve CPU implementations for correctness and tests.

## Paged Attention Metadata

`PagedAttentionRuntimeMetadata` contains:

- `slot_mapping`
- `seq_indices`
- `context_lens`
- `block_tables`
- `host_block_tables` and `host_block_table_count`
- `query_segments` and `query_segment_count`
- step-scoped `scratch`
- `block_size_tokens`
- `query_segments_valid`
- `enabled`

`LlamaAttentionParams` contains the tensor pointers, execution context, metadata pointer, layer ID, and attention dimensions needed for one attention call.

## Paged Attention Interfaces

- `llama_attention_forward(const LlamaAttentionParams& params)`: primary model path.
- `llama_attention(...)`: convenience wrapper that builds `LlamaAttentionParams`.

Paged KV metadata is passed explicitly through `LlamaAttentionParams`; the operator does not depend on thread-local state.

## Attention Behavior

The attention path validates:

- tensor dtype and shape;
- attention head divisibility;
- device consistency;
- KV cache device matching;
- KV block byte size;
- rank-3 block table shape;
- valid layer and sequence indices.

For paged attention, keys and values are written into the physical KV block for each scheduled token, then causal attention reads previous positions through `block_tables`.

The CPU paged backend is straightforward and correctness-oriented. FP32 CUDA attention uses custom kernels for decode and small query segments and segmented SDPA for sufficiently large full or chunked-prefill segments. Chunked prefill gathers its cached prefix and uses an offset-causal mask. BF16 KV uses its dedicated paged CUDA kernel; unsupported dispatches retain torch-backed reference bridges.

## Runtime Metadata Direction

The current model path passes attention metadata explicitly through `RuntimeContext`. New operators must not introduce process-wide or thread-local request metadata.
