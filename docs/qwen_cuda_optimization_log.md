# Qwen CUDA Optimization Log

This document records the Qwen2.5 CUDA correctness and performance optimization work for TinyLLM. The main benchmark target is `/models/Qwen2.5-1.5B-Instruct` with greedy generation aligned against Hugging Face Transformers.

## Baseline Problem

The initial quick benchmark had a correctness mismatch in `quick_batch`: the second TinyLLM batch request diverged from the Transformers output. The remaining performance profile after correctness work showed three main issues:

- `repetition_penalty` triggered a CPU sampler fallback and copied full logits from CUDA to CPU.
- The default `kv_num_blocks=256` was too small for Qwen quick batch, causing preemption and full-context recompute.
- Runtime profiling counted generated-token replay as prefill, so `decode_ms_total` and `avg_decode_tokens` were reported as zero.

Representative pre-optimization `quick_batch` metrics:

| Metric | Before optimization |
| --- | ---: |
| `avg_total_latency_ms` | `1532.822` |
| `avg_first_token_latency_ms` | `268.485` |
| `sampling_ms` | `939.587` |
| `prefill_ms` | `583.367` |
| `decode_ms_total` | `0.000` |
| `avg_prefill_tokens` | `1022.0` |
| `avg_decode_tokens` | `0.0` |

The expected `quick_batch` second output after correctness alignment is:

```text
[7388, 448, 28765, 12282, 323, 33773, 11, 1293]
```

## Optimization 1: Scheduler Preemption Correctness

### Optimization Point

Preempted requests were recomputed from the original prompt only. If a request had already generated tokens, the next recompute discarded generated context and could sample the same prompt-final token again.

### Strategy

Use the full request context, `prompt + generated tokens`, when scheduling recompute after preemption. Keep the scheduler accounting rule that the prefill-final row produces the first generated token, and decode rows consume the last generated token.

### Files

- `src/runtime/scheduler.cpp`
- `tests/unit/test_scheduler.cpp`

### Result

The `quick_batch` second request now matches Transformers:

```text
[7388, 448, 28765, 12282, 323, 33773, 11, 1293]
```

This was a correctness fix, not a direct performance optimization. It made later benchmark numbers meaningful because TinyLLM and Transformers were generating the same tokens.

## Optimization 2: Generation Config and Repetition Penalty Correctness

### Optimization Point

Qwen's `generation_config.json` sets `repetition_penalty=1.1`. Without honoring it, TinyLLM could not match HF `generate(do_sample=False)` defaults. The initial implementation also treated `0 < repetition_penalty < 1.0` as inactive.

### Strategy

Load `generation_config.json` in the benchmark and generation tools, propagate `repetition_penalty` into sampling metadata, and apply HF-compatible repetition penalty semantics for any positive penalty different from `1.0`.

### Files

- `include/tiny_llm/runtime/generation_config.h`
- `src/runtime/generation_config.cpp`
- `src/runtime/sampler.cpp`
- `tests/unit/test_sampler.cpp`
- `tools/llama_engine_generate.cpp`
- `benchmark/llama_engine_benchmark.cpp`

### Result

The Qwen quick correctness baseline passed for `quick_interactive` and `quick_batch`. The cost was a large sampler bottleneck because the first implementation copied full CUDA logits to CPU whenever repetition penalty was active.

## Optimization 3: CUDA Repetition Penalty Sampler

### Optimization Point

With Qwen `repetition_penalty=1.1`, sampling dominated `quick_batch` runtime. A control run without `generation_config.json` reduced `sampling_ms` from about `939.587ms` to about `92.796ms`, showing that the penalty CPU fallback was the first performance bottleneck.

### Strategy

Keep logits on GPU for CUDA sampling. For rows with active repetition penalty, clone only the selected row, adjust history-token logits on GPU, run GPU `argmax`, and copy only final token IDs back to CPU. This keeps HF semantics while avoiding full-logit CPU transfer.

### Files

- `src/runtime/sampler.cpp`
- `tests/unit/test_sampler.cpp`

### Before/After

| Metric, quick_batch | Before CUDA sampler | After CUDA sampler/KV/profiling phase |
| --- | ---: | ---: |
| `avg_total_latency_ms` | `1532.822` | `506.001` |
| `sampling_ms` | `939.587` | `63.887` |
| `avg_prefill_tokens` | `1022.0` | `128.0` |
| `avg_decode_tokens` | `0.0` | `14.0` |

The CUDA sampler path is covered by a CPU-vs-CUDA repetition penalty test with non-dense rows and both `penalty > 1` and `penalty < 1`.

## Optimization 4: KV Block Auto-Sizing

### Optimization Point

The Qwen quick batch case needs more than the default 256 KV blocks. With block size 16 and 28 layers, two sequences of roughly 72 tokens need about `2 * ceil(72 / 16) * 28 = 280` blocks before slack. The old default caused repeated preemption/recompute.

### Strategy

Add `--kv-num-blocks` to benchmark and generation tools. When not explicitly supplied, estimate the required block count from per-prompt token counts, `max_new_tokens`, block size, and layer count, then add slack and keep a floor of 256.

### Files

- `benchmark/llama_engine_benchmark.cpp`
- `tools/llama_engine_generate.cpp`

### Before/After

| Metric, quick_batch | Before auto-sizing | After auto-sizing |
| --- | ---: | ---: |
| `kv_num_blocks` | `256` | `336` |
| `avg_prefill_tokens` | `1022.0` | `128.0` |

The prefill token count now reflects the actual prompt workload instead of repeated recompute.

## Optimization 5: Profiling Classification Fix

### Optimization Point

Generated-token replay during recompute was counted as prefill because profiling used scheduler chunk type rather than token positions. This produced `decode_ms_total=0` and `avg_decode_tokens=0` even though decode work occurred.

### Strategy

Carry `prompt_token_count` in `RequestData`. In `ModelRunner`, classify each scheduled token by position:

```text
position < prompt_token_count  -> prefill
position >= prompt_token_count -> decode/replay
```

### Files

- `include/tiny_llm/runtime/scheduler.h`
- `src/runtime/scheduler.cpp`
- `src/runtime/model_runner.cpp`

### Before/After

| Metric, quick_batch | Before classification fix | After classification fix |
| --- | ---: | ---: |
| `decode_ms_total` | `0.000` | `206.681` |
| `avg_decode_tokens` | `0.0` | `14.0` |

This made later bottleneck analysis actionable.

## Optimization 6: RoPE Frequency Cache

### Optimization Point

The profile-detail run after sampler and KV fixes showed RoPE as one of the remaining model-internal hotspots. The previous path rebuilt the inverse-frequency tensor during repeated layer calls, adding avoidable tensor construction overhead.

### Strategy

Cache the RoPE `inv_freq` tensor per device inside `RotaryEmbedding`, and add an `apply_rope` overload that accepts the precomputed tensor. This keeps the public operator path available while allowing the LLaMA/Qwen runtime layer path to reuse the cached frequency tensor.

### Files

- `include/tiny_llm/models/modules/rotary_embedding.h`
- `src/models/modules/rotary_embedding.cpp`
- `include/tiny_llm/operators/llama_ops.h`
- `src/operators/llama_ops.cpp`

### Before/After

| Metric, profile-detail quick_batch | Before RoPE cache | After RoPE/QKV phase |
| --- | ---: | ---: |
| `rope_ms` | `~84.6` | `85.681` |

The profile did not show a stable RoPE latency reduction after this change. The practical value is that frequency construction is no longer repeated, but the remaining RoPE cost is dominated by per-step `cos/sin` and tensor application work. A later optimization should consider caching position-specific `cos/sin` values across layers.

## Optimization 7: QKV Stacked Linear Cache

### Optimization Point

QKV projection was the largest remaining model-internal hotspot after sampler and KV fixes. The old stacked linear path still executed separate q/k/v matmuls, which is expensive for every layer and decode step.

### Strategy

When `Linear::bind_stacked_weights()` receives q/k/v weights, build a concatenated weight and bias cache. In CUDA forward, use one matmul against the stacked weight and then split the result into q/k/v views. Keep the original separate path available as a fallback.

### Files

- `include/tiny_llm/models/modules/linear.h`
- `src/models/modules/linear.cpp`
- `tests/unit/test_linear_module.cpp`

### Before/After

| Metric, profile-detail quick_batch | Before QKV cache | After QKV cache |
| --- | ---: | ---: |
| `qkv_proj_ms` | `~103` to `149` | `84.372` |

This reduced QKV projection time in the synchronized profile. The tradeoff is an extra cached copy of stacked QKV weights and bias, so model load memory and GPU memory usage increase.

## End-State Performance Summary

The best current quick benchmark report is:

```text
benchmark/results/perf_after_rope_qkv_cache_20260531_001505.json
```

The best current profile-detail report is:

```text
/tmp/tinyllm_profile_results/profile_after_rope_qkv_cache_20260531_001612.json
```

### quick_batch

| Metric | Initial bottleneck baseline | After sampler/KV/profiling | After RoPE/QKV |
| --- | ---: | ---: | ---: |
| `avg_total_latency_ms` | `1532.822` | `506.001` | `395.097` |
| `avg_first_token_latency_ms` | `268.485` | `not captured here` | `205.867` |
| `sampling_ms` | `939.587` | `63.887` | `30.393` |
| `prefill_ms` | `583.367` | `not captured here` | `175.714` |
| `decode_ms_total` | `0.000` | `206.681` | `184.396` |
| `avg_prefill_tokens` | `1022.0` | `128.0` | `128.0` |
| `avg_decode_tokens` | `0.0` | `14.0` | `14.0` |
| `kv_num_blocks` | `256` | `336` | `336` |

### quick_interactive

| Metric | After RoPE/QKV |
| --- | ---: |
| `avg_total_latency_ms` | `303.450` |
| `avg_first_token_latency_ms` | `154.185` |
| `sampling_ms` | `22.913` |

## Verification Performed

The following checks passed after the second optimization stage:

```bash
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure -R 'Linear|Sampler|Scheduler|KVCache|ModelRunner|EngineCore|PagedAttention|LlamaOps'
cmake --build build -j
git diff --check
```

The CUDA test set reported `35/35` passing. Additional Qwen checks showed token-level agreement with Transformers for quick benchmark, batch=4 OSL=16, long prefill OSL=8, and batch=8 OSL=8.

## Remaining Performance Work

- Replace the libtorch reference attention path with a real optimized paged-attention CUDA kernel.
- Cache RoPE `cos/sin` values across layers and positions instead of only caching `inv_freq`.
- Measure memory impact from stacked QKV caches before enabling it for larger models by default.
- Run `focus`, `regression`, and eventually `full` benchmark presets with timeouts after the attention path is improved.
- Keep `profile-detail` numbers separate from headline throughput because they include extra CUDA synchronizations.

## Optimization 8: Decode Attention, RoPE Cache, and Memory Observability

### Optimization Point

After the QKV cache stage, profile-detail still showed `quick_batch attention_ms=81.673` and `rope_ms=85.681`. The stacked QKV cache also needed an explicit memory fallback path for larger models.

### Strategy

Add a non-atomic CUDA paged-attention branch for small-row decode batches while keeping the old atomic path for multi-token prefill. This avoids the correctness risk found when using the short path for long prefill rows. Extend RoPE from `inv_freq` caching to a shared per-device `cos/sin` cache used by the CUDA runtime path. Add `TINYLLM_QKV_STACKED_CACHE=0/1` to disable stacked QKV cache construction and add CUDA allocator memory fields to benchmark JSON.

### Files

- `src/operators/paged_attention/paged_attention_kernels.cu`
- `src/models/modules/rotary_embedding.cpp`
- `src/operators/llama_ops.cpp`
- `src/models/modules/linear.cpp`
- `benchmark/llama_engine_benchmark.cpp`

### Before/After

| Metric, quick_batch | After RoPE/QKV baseline | After attention/RoPE/memory |
| --- | ---: | ---: |
| `avg_total_latency_ms` | `395.097` | `331.383` |
| `avg_first_token_latency_ms` | `205.867` | `203.586` |
| `sampling_ms` | `30.393` | `27.655` |
| `decode_ms_total` | `184.396` | `123.884` |
| `cuda_memory_allocated_mb` | `not reported` | `6292.897` |
| `cuda_memory_reserved_mb` | `not reported` | `6580.000` |
| `cuda_memory_peak_allocated_mb` | `not reported` | `6441.275` |

| Metric, profile-detail quick_batch | After RoPE/QKV baseline | After attention/RoPE/memory |
| --- | ---: | ---: |
| `attention_ms` | `81.673` | `22.831` |
| `rope_ms` | `85.681` | `82.603` |
| `qkv_proj_ms` | `84.372` | `79.041` |
| `mlp_ms` | `67.935` | `67.552` |

The current quick report is:

```text
benchmark/results/perf_after_attention_rope_memory_20260531_005009.json
```

The current profile-detail report is:

```text
/tmp/tinyllm_profile_results/profile_after_attention_rope_memory_20260531_005114.json
```

The QKV cache disabled smoke report is:

```text
/tmp/tinyllm_qkv_cache_off_results/qkv_cache_off_quick_20260531_005216.json
```

### Regression Gate

The regression preset completed under the 900 second timeout:

```text
benchmark/results/regression_after_cuda_perf_20260531_005321.json
```

| Workload | `avg_total_latency_ms` | `avg_first_token_latency_ms` | `decode_ms_total` | `sampling_ms` |
| --- | ---: | ---: | ---: | ---: |
| `interactive` | `750.826` | `32.422` | `692.407` | `13.415` |
| `chat_serving` | `2098.704` | `57.285` | `1641.915` | `135.575` |

### Notes

- The optimized attention branch is intentionally limited to `rows <= 8` to target decode and small batched decode. Long prefill remains on the older atomic fallback because the non-atomic branch did not match the torch reference for Qwen long prefill.
- `TINYLLM_QKV_STACKED_CACHE=0` preserves token correctness and provides a lower-memory fallback, with expected latency regression.

### Commit References

- `d5608ab perf: optimize short cuda paged attention`
- `754b2aa perf: cache rope cos sin tables`
- `e29328e perf: add qkv cache control and cuda memory metrics`
- `cb883a8 docs: update qwen cuda optimization results`

## Optimization 9: RoPE Apply Kernel and MLP Projection Fusion

### Optimization Point

The previous stage proved that the short attention fast path helped quick decode, but quick was too small to represent steady serving behavior. A follow-up `regression --profile-detail` and a TinyLLM-only `decode_heavy` profile showed two different regimes:

- `chat_serving`: MLP and attention dominate; RoPE is no longer the largest bucket after cached cos/sin.
- `decode_heavy`: attention dominates long decode, followed by MLP; RoPE is a smaller but still measurable cost.

### Strategy

Replace the cached RoPE CUDA application path with a dedicated kernel. The old cache avoided rebuilding `cos/sin`, but still used libtorch `index_select`, narrow views, clones, and `copy_`. The new kernel rotates q/k in place from `positions`, `cos_cache`, and `sin_cache`.

Fuse the MLP gate/up projections through the existing stacked `Linear` path. This turns gate and up into one combined projection output, then applies a dedicated CUDA `silu_multiply` kernel before the down projection. The public model API is unchanged, and `llama_tensor_dump` was updated to keep dumping `mlp_gate` and `mlp_up`.

### Files

- `src/operators/llama_ops_kernels.cu`
- `src/operators/llama_ops.cpp`
- `include/tiny_llm/models/llama_decoder_layer.h`
- `src/models/llama_layer.cpp`
- `src/models/llama_model.cpp`
- `tools/llama_tensor_dump.cpp`
- `tests/unit/test_llama_ops.cpp`

### Before/After

| Metric, quick_batch | After attention/RoPE/memory | After RoPE kernel + MLP fuse |
| --- | ---: | ---: |
| `avg_total_latency_ms` | `331.383` | `311.352` |
| `avg_first_token_latency_ms` | `203.586` | `186.749` |
| `sampling_ms` | `27.655` | `28.725` |
| `decode_ms_total` | `123.884` | `119.895` |

| Metric, profile-detail quick_batch | After attention/RoPE/memory | After RoPE kernel |
| --- | ---: | ---: |
| `attention_ms` | `22.831` | `22.618` |
| `rope_ms` | `82.603` | `45.154` |
| `qkv_proj_ms` | `79.041` | `83.349` |
| `mlp_ms` | `67.552` | `78.818` |

The quick profile after MLP fusion was not preserved because later profile runs reused the same output directory. The preserved quick correctness report is:

```text
benchmark/results/quick_after_mlp_fuse_20260531_014536.json
```

### Serving-Like Profile

| Workload, profile-detail | Before MLP fuse `mlp_ms` | After MLP fuse `mlp_ms` | After total latency |
| --- | ---: | ---: | ---: |
| `interactive` | `402.905` | `370.906` | `828.972` |
| `chat_serving` | `911.199` | `891.667` | `2262.529` |

The preserved regression profile is:

```text
/tmp/tinyllm_regression_profile_mlp/regression_profile_after_mlp_fuse_20260531_014934.json
```

### Regression Gate

The non-profile regression gate completed under the 900 second timeout:

```text
benchmark/results/regression_after_next_perf_20260531_015332.json
```

| Workload | Previous regression total | After RoPE kernel + MLP fuse total | `decode_ms_total` | `decode_ms_per_token` |
| --- | ---: | ---: | ---: | ---: |
| `interactive` | `750.826` | `649.746` | `587.789` | `9.330` |
| `chat_serving` | `2098.704` | `1889.914` | `1446.261` | `1.423` |

### Decode-Heavy Profile

The TinyLLM-only `decode_heavy` profile confirms that long decode is now attention-bound:

| Workload | `avg_total_latency_ms` | `attention_ms` | `mlp_ms` | `rope_ms` | `qkv_proj_ms` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decode_heavy` | `7751.625` | `3979.134` | `1716.856` | `104.928` | `429.916` |

The preserved decode-heavy profile is:

```text
/tmp/tinyllm_profile_results/decode_heavy_profile_after_mlp_fuse_20260531_014821.json
```

### Notes

- The RoPE kernel is the clearer win in this stage; it cuts quick profile `rope_ms` from `82.603` to `45.154`.
- MLP fusion is modest in serving-like profile and not the next primary target. Long decode remains dominated by paged attention.
