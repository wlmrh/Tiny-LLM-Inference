# Tokenizer and Processors Module

This module handles the boundary between user strings and engine token IDs.

## Main Files

- `include/tiny_llm/runtime/tokenizer.h`
- `src/runtime/tokenizer.cpp`
- `include/tiny_llm/runtime/processors.h`
- `src/runtime/processors.cpp`

## `Tokenizer`

`Tokenizer` is the runtime tokenizer interface.

Interfaces:

- `encode(text)`: convert text to token IDs.
- `decode(ids)`: convert token IDs to text.
- `vocab_size()`: tokenizer/model vocabulary size used for validation.
- `bos_id()`, `eos_id()`, `unk_id()`: special token IDs.
- `is_fixed_vocab()`: true when valid token IDs cannot grow dynamically.
- `is_valid_token_id(id)`: token ID validation.

## `TokenizerRegistry`

`TokenizerRegistry` is a small dependency-injection holder.

Attribute:

- `tokenizer_`: non-owning tokenizer pointer.

Interfaces:

- `mutable_tokenizer()`
- `tokenizer()`

## `HFLlamaTokenizer`

`HFLlamaTokenizer` implements `Tokenizer` for HuggingFace LLaMA/Qwen-compatible tokenizers.

Important behavior:

- Loads `tokenizer.json` through `tokenizers::Tokenizer::FromBlobJSON` when present.
- Falls back to `tokenizer.model` through `FromBlobSentencePiece`.
- Loads `config.json` and optional tokenizer metadata files to resolve special tokens.
- Accepts special tokens represented as strings, JSON objects with `content`, or `null`.
- Treats `unk_token_id` as optional.
- Uses `max(tokenizer_vocab, config.vocab_size)` as runtime vocab size so padded model vocabularies are accepted.
- If tokenization returns no IDs and BOS exists, `encode()` returns a BOS-only prompt.

Attributes:

- `impl_`: private tokenizers-cpp handle and metadata.
- `bos_id_`, `eos_id_`, `unk_id_`: resolved special token IDs.

Interfaces:

- `from_model_dir(hf_model_dir)`: construct tokenizer from HF directory.
- Standard `Tokenizer` methods.

## `InputPreprocessor`

`InputPreprocessor` converts user prompt inputs into `EngineCoreRequest`.

Attributes:

- `tokenizer_`: non-owning tokenizer pointer.
- `model_`: currently unused non-owning model pointer slot.
- `default_max_tokens_`: fallback max generation length.
- `next_internal_id_`: monotonically increasing internal ID.
- `external_to_internal_id_`: duplicate external ID guard.

Interfaces:

- `process_inputs(prompt, user_params, ext_request_id)`: returns a validated `EngineCoreRequest`.
- `release_request(external_id, internal_id)`: releases an active external ID after request completion.

Internal steps:

1. Assign a new internal ID.
2. Bind or synthesize an external ID.
3. Apply chat template. Current implementation is a no-op.
4. Tokenize the prompt.
5. Normalize sampling parameters.
6. Validate prompt tokens and sampling settings.

Sampling normalization copies user fields and appends tokenizer EOS to `stop_token_ids` when missing.

## `OutPreprocessor`

`OutPreprocessor` owns user-facing output state.

Attributes:

- `tokenizer_`: non-owning tokenizer pointer.
- `states_`: map from internal ID to `RequestState`.

Interfaces:

- `add_request(const EngineCoreRequest&)`: registers output state and decodes the initial prompt.
- `process_outputs(core_outputs)`: converts sampled token IDs to `UserOutput`.
- `has_unfinished_requests()`: checks frontend states.

Internal behavior:

- `incremental_decode()` decodes prompt plus generated tokens and returns only the newly decoded suffix.
- `check_stop_criteria()` finishes on EOS, configured stop token, or `max_tokens`.

## Output Types

`UserOutput` contains:

- `internal_id`
- `external_id`
- `delta_text`
- `text`
- `generated_token_ids`
- `is_finished`
- `finish_reason`
- `error_message`

`LLM` converts this into `CompletionOutput` and `CompletionStreamOutput`. `CompletionStreamOutput` extends the final output shape with stream-only fields such as `prompt_index`, `delta_text`, and latest `token_id`.
