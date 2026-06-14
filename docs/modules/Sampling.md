# Sampling Module

The sampling module implements greedy token selection and seeded non-greedy sampling with HuggingFace-style repetition penalty.

## Main Files

- `include/tiny_llm/runtime/sampler.h`
- `src/runtime/sampler.cpp`
- `include/tiny_llm/runtime/generation_config.h`
- `src/runtime/generation_config.cpp`

## Sampling Parameters

`SamplingParamsCommon` stores fields shared by user-facing and normalized engine parameters. `UserSamplingParams` is the user-facing structure and `SamplingParams` is the normalized engine structure.

Shared attributes:

- `temperature`
- `top_p`
- `top_k`
- `repetition_penalty`
- `seed`
- `ignore_eos`
- `stop_token_ids`

`UserSamplingParams::max_tokens` uses `0` as the default sentinel, which means "use the runtime default". Negative values are invalid. `SamplingParams::max_tokens` is always a positive normalized value after input preprocessing.

Current execution behavior:

- `temperature == 0` uses greedy selection.
- `temperature > 0` uses seeded multinomial sampling.
- `top_k` and `top_p` filter logits before sampling.
- `repetition_penalty` affects logits before filtering.
- `max_tokens` affects scheduler/frontend stopping.
- `stop_token_ids` affect scheduler/frontend stopping.

Prefer default construction plus field assignment for public sampling params instead of positional aggregate initialization.

## Interfaces

- `sample_greedy_rows(logits, sample_rows, vocab_size, token_histories, sampling_params, request_ids)`: samples selected rows from a logits tensor. The function name is historical; it also handles configured seeded non-greedy sampling.

`sample_greedy_rows()` returns a vector with length equal to `logits.size(0)`. Non-sampled rows are `-1`.

## CPU Path

The CPU path:

1. Validates logits dtype and shape.
2. Copies/contiguizes logits on CPU if needed.
3. For each requested row, applies repetition penalty to unique history tokens when active.
4. Applies top-k and top-p filters when configured.
5. Returns argmax token IDs for greedy rows or seeded multinomial samples for non-greedy rows.

History token IDs are validated against the model vocabulary.

## CUDA Path

The CUDA greedy path keeps logits on the GPU:

- Without repetition penalty, it uses `argmax(dim=1)` for dense sampled rows or selects sampled rows first for non-dense rows.
- With repetition penalty, it builds row/history metadata in reusable thread-local CUDA scratch tensors, launches CUDA kernels to mark repetition history and run argmax with penalty adjustment, and copies only sampled token IDs back to CPU.

This avoids full-logit CPU transfer for Qwen generation configs that use repetition penalty. Non-greedy CUDA sampling currently uses the shared CPU sampling path for correctness and deterministic behavior.

## Generation Config

`GenerationConfig` currently contains:

- `repetition_penalty`

Interfaces:

- `load_generation_config_from_dir(model_dir)`: loads `generation_config.json` when present, otherwise returns defaults.

The generation tools and benchmarks can use this config to match HuggingFace greedy defaults.

## Determinism

Sampling is deterministic for the same `seed`, request ID, token history length, and sample index. Default seed `0` is still deterministic; callers can set another seed to choose a different reproducible sample stream.
