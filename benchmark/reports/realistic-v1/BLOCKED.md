# TinyLLM Realistic Benchmark v1: Blocked Run

Status: **blocked; no performance report is published**

The realistic-v1 run stopped at the first deterministic TinyLLM runtime failure, as required by the
benchmark gate. No workload was substituted, no partial performance result is presented as a completed
report, and no runtime implementation was changed during this benchmark work.

## Candidate and environment

- Runtime candidate: `15672b369bafe93d1544649e72e2fee141821611`
- Candidate worktree: `git_dirty=false`
- GPU: NVIDIA GeForce RTX 4080 SUPER 32 GB
- Driver: 595.71.05
- CUDA toolkit: 12.8.93
- Suite environment: Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 4.55.2
- vLLM environment: vLLM 0.25.1, PyTorch 2.11.0
- Model: Qwen/Qwen2.5-1.5B-Instruct at revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- TinyLLM compute / KV dtype: FP32 / FP32

## Fixed data

- BurstGPT v2.0, commit `7eb2c4f8350f8a6985272386f5c14af1f678b299`
  - `BurstGPT_without_fails_3.csv`
  - 217,312,026 bytes
  - SHA-256 `3326259f9efb11845bc5ef85fa97e6f691050b0974621c91ef22acd566c43a40`
- OpenAssistant/oasst1 revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`
  - `2023-04-12_oasst_ready.trees.jsonl.gz`
  - 34,145,252 bytes
  - SHA-256 `2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7`

Preparation produced 32,372 eligible OASST1 prefixes, 4,771,985 usable BurstGPT rows, and three
non-overlapping 1,000-request windows. The public selection metadata contains no prompt text, OASST user
ID, or raw BurstGPT session ID.

## Gates completed before the failure

- Python benchmark tests: 18/18 passed on the runtime candidate.
- CPU CTest: 65/65 passed; four explicitly model-backed SmolLM2 tests also passed without skips.
- CUDA CTest: 78/78 passed, including model alignment and CUDA generation smoke tests.
- Qwen mixed-OSL smoke: 32/32 requests completed; 837 requested tokens matched 837 token events.
- Three-backend correctness: 8/8 requests had exact pairwise token-ID agreement among TinyLLM,
  Transformers, and vLLM.
- Completed performance shards without backend skip, error, or OOM:
  - short chat: 3/3
  - medium chat: 3/3
  - long prefill: 1/3

These partial performance measurements are retained only as diagnostic artifacts and are not published as
benchmark results.

## Blocking failure

The second long-prefill shard failed in TinyLLM before the Transformers and vLLM measurements were
accepted:

```text
llama_engine_benchmark failed: OutPreprocessor::incremental_decode:
decoded prefix length is invalid.
```

The failing request is identified without publishing its text:

- request ID: `long_prefill-02-002`
- Qwen input length: 1,578 tokens
- requested output length: 64 tokens, fixed output / ignore EOS
- prompt SHA-256: `90bd672f3f8a3932c8e6496e31e4f58699dd78bd392244215701deab3048fef0`

Isolation results:

- The other three requests in the shard pass when run individually.
- The failing request passes for output lengths 1 through 14.
- It fails deterministically beginning at output token 15.
- Output length 64 reproduced the same failure twice.
- The failure is not an OOM and also occurs with a single request and no warmup.

## Consequence and next action

The remaining two long-prefill shards, all long-decode shards, three capacity calibrations, and twelve
trace replays were not run. `trace_replay.json`, `offline.json`, and a realistic-v1 headline report must not
be published from this incomplete run.

The next step requires a separately authorized runtime investigation of incremental token decoding in
`OutPreprocessor`. After a fix, the entire correctness, offline, capacity, and trace sequence must be rerun
from a new clean candidate commit; the current partial numbers must not be carried forward.
