# Request State Module

The request-state module defines the lifecycle and token accounting used by the scheduler and frontend output processor.

## Main Files

- `include/tiny_llm/runtime/request.h`
- `src/runtime/request.cpp`
- `include/tiny_llm/runtime/processors.h`
- `src/runtime/processors.cpp`

## Scheduler Request State

`Request` is the scheduler-owned state object.

Attributes:

- `request_id`: internal numeric request ID.
- `priority`: reserved for future scheduling policies.
- `sampling_params`: normalized `SamplingParams`.
- `status`: `WAITING`, `RUNNING`, or `FINISHED`.
- `num_computed`: number of tokens whose KV cache has been computed.
- `prompt_token_ids`: immutable prompt tokens.
- `_all_token_ids`: full context, initialized to prompt tokens and extended with generated tokens.

Interfaces:

- `reset_generated_tokens()`: resets `_all_token_ids` to the prompt.
- `generated_tokens()`: returns `_all_token_ids.size() - prompt_token_ids.size()`.
- `has_valid_token_layout()`: checks that `_all_token_ids` begins with the prompt.

## Lifecycle

```text
WAITING
  -> RUNNING after KV slots are allocated and the request is scheduled
  -> WAITING if preempted
  -> FINISHED after stop-token or max-length condition
  -> erased after cleanup
```

When a request is preempted, `_all_token_ids` is not reset. Only `num_computed` is reset, so the scheduler recomputes prompt plus generated context.

## Frontend Request State

`RequestState` is owned by `OutPreprocessor`, not by `Scheduler`.

Attributes:

- `internal_id`: matches the scheduler/core request ID.
- `external_id`: user-visible ID.
- `sampling_params`: normalized sampling settings.
- `prompt_token_ids`: prompt tokens for full-context decoding.
- `generated_token_ids`: generated token sequence.
- `decoded_prefix_len`: byte length of the decoded prompt plus prior tokens.
- `is_finished`: frontend completion flag.
- `finish_reason`: `eos`, `stop_token`, `length`, or `error`.
- `cached_text`: decoded prompt plus generated text.

## Output Semantics

`OutPreprocessor::incremental_decode()` appends a sampled token, decodes `prompt + generated`, and returns only the suffix after `decoded_prefix_len`.

`OutPreprocessor::check_stop_criteria()` finishes on:

- tokenizer EOS ID;
- any normalized stop token ID;
- generated token count reaching `max_tokens`.

The scheduler also checks stop token and length. The frontend check determines user-visible finish reason and output state.
