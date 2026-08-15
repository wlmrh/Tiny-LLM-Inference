# KV Cache and Memory Module

This module provides persistent paged KV cache storage and short-lived per-step workspace memory.

## Main Files

- `include/tiny_llm/runtime/kv_cache.h`
- `src/runtime/kv_cache.cpp`
- `include/tiny_llm/runtime/kv_cache_manager.h`
- `src/runtime/scheduler.cpp`
- `include/tiny_llm/core/allocator.h`
- `src/core/allocator_common.cpp`
- `src/core/cpu/cpu_allocator.cpp`
- `src/core/cuda/cuda_allocator.cpp`

## `StackAllocator`

`StackAllocator` is a monotonic workspace allocator for temporary tensors.

Attributes:

- `base_ptr_`: base pointer of the contiguous pool.
- `total_size_`: capacity in bytes.
- `offset_`: current bump pointer offset.
- `parallel_config_`: CPU/CUDA backing device.
- `peak_offset_`: optional debug high-water mark when debug stats are enabled.

Interfaces:

- `StackAllocator(size_t pool_size)`: CPU workspace.
- `StackAllocator(size_t pool_size, ParallelConfig)`: workspace on the selected device.
- `reset()`: resets the bump offset to zero.
- `allocate(size_t bytes)`: reserves aligned bytes and returns a raw pointer.
- `make_tensor(shape, dtype)`: returns a non-owning torch tensor view over allocated workspace memory.
- `parallel_config()` / `device()`: return backing device information.

Workspace tensors are invalid after the next `reset()` or `ExecutionContext::begin_step()`.

## `BlockAllocator`

`BlockAllocator` manages fixed-size persistent KV blocks.

Attributes:

- `memory_pool_`: raw base pointer.
- `num_blocks_`: total block count.
- `block_size_`: byte size per block.
- `parallel_config_`: CPU/CUDA backing device.
- `free_list_`: LIFO list of available physical block IDs.
- `allocated_`: allocation-state flags used to detect invalid or duplicate frees.

Interfaces:

- `BlockAllocator(num_blocks, block_size_bytes, memory_pool, ParallelConfig)`: binds an external block pool on an explicit device.
- `allocate_block()`: returns a block ID or `-1` when exhausted.
- `free_block(int32_t block_id)`: returns a block to the free list.
- `get_block_ptr(int32_t block_id)`: maps block ID to raw pointer.
- `free_block_count()`, `total_block_count()`, `block_size_bytes()`, `memory_pool()`: pool metadata.

KV blocks persist across engine steps and are released only when a sequence ends or is preempted. Invalid and duplicate frees are rejected.

## `KVCache`

`KVCache` manages sequence-to-block metadata. It does not own model logic and does not perform attention by itself.

Configuration:

- `num_layers`: transformer layer count.
- `block_size_tokens`: token capacity per physical block.

Attributes:

- `cfg_`: static cache configuration.
- `owned_blocks_`: optional owned `BlockAllocator`.
- `blocks_`: active block allocator.
- `parallel_config_`: backing device.
- `seqs_`: map from sequence ID to per-layer page tables.

Interfaces:

- `KVCache(Config, BlockAllocator*)`: binds an existing block allocator.
- `start_sequence(seq_id)`: initializes per-layer page tables and rejects duplicate active sequence IDs.
- `end_sequence(seq_id)`: releases all blocks used by a sequence.
- `ensure_capacity(seq_id, layer_id, token_pos)`: allocates blocks so `token_pos` can be stored.
- `page_table(seq_id, layer_id)`: returns logical-to-physical block IDs.
- `free_block_count()`, `total_block_count()`, `block_size_bytes()`, `block_pool_base()`, `block_ptr()`: backing storage metadata.
- `parallel_config()` / `device()`: backing device.

## `KVCacheManager`

`KVCacheManager` is a scheduler helper that binds an external `KVCache` or owns the runtime allocator/cache pair and exposes scheduling-oriented operations.

Attributes:

- `owned_kv_`: optional owned KV cache.
- `kv_`: active KV cache pointer.

Interfaces:

- `bind(KVCache*)`: binds external cache.
- `init_owned(..., ParallelConfig)`: constructs an owned `BlockAllocator` and `KVCache` from raw pool parameters on an explicit device.
- `start_sequence(core_seq_id)` / `end_sequence(core_seq_id)`: sequence lifecycle.
- `estimate_append_new_blocks(...)`: estimate blocks needed for one decode append.
- `estimate_prefill_new_blocks(...)`: estimate blocks needed for a prefill chunk.
- `allocate_slots(...)`: ensure enough blocks exist for a scheduled token range.
- `refresh_block_tables(...)`: all-layer block tables used by `ModelRunner`.
- `free_block_count()`, `num_layers()`, `kv_cache()`: metadata and bound cache access.

`allocate_slots()` performs a capacity check across all layers before allocating. This prevents partial allocation when free blocks are insufficient.

## KV Block Layout

Each physical block stores keys followed by values in the configured runtime KV dtype:

```text
K region: block_size_tokens * (num_key_value_heads * head_dim) elements
V region: block_size_tokens * (num_key_value_heads * head_dim) elements
```

The required block byte size is:

```text
2 * block_size_tokens * (num_key_value_heads * head_dim)
  * runtime_dtype_size(kv_cache_dtype)
```

FP32 elements use four bytes and BF16 elements use two bytes; block sizing must not assume `sizeof(float)`.
