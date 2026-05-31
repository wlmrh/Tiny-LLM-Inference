# HF SafeTensor Integration Plan (for trl-internal-testing/tiny-random-LlamaForCausalLM)

## 1. Objective

Enable the current Tiny-LLM-Inference runtime to load and run a HuggingFace Llama model from SafeTensor artifacts.

Target local model directory:

- /Users/tangqi/weights

Observed local files:

- config.json
- generation_config.json
- model.safetensors
- tokenizer.json
- tokenizer.model
- tokenizer_config.json
- special_tokens_map.json
- pytorch_model.bin (fallback reference only)

Model config snapshot (from `/Users/tangqi/weights/config.json`):

- architectures: LlamaForCausalLM
- num_hidden_layers: 2
- hidden_size: 16
- intermediate_size: 64
- num_attention_heads: 4
- hidden_act: silu
- rms_norm_eps: 1e-6
- vocab_size: 32000
- bos_token_id: 0
- eos_token_id: 1

Tokenizer snapshot (from tokenizer files):

- tokenizer_class: LlamaTokenizer
- model_max_length: 2048
- special tokens include <unk>, <s>, </s>

Important risk to resolve early:

- `config.json` token ids may not perfectly align with tokenizer-derived ids in all HF exports.
- This must be normalized in Phase 0/Phase 2 as a strict contract.

## 2. Historical Gap

1. Runtime needed direct HuggingFace Llama-family weight execution.
2. Runtime needed a SafeTensor parser in C++.
3. Runtime tokenizer needed HuggingFace tokenizer behavior.
4. Model construction needed to accept a HF directory.

Therefore, this should be delivered in phases, not one-shot.

## 3. Phased Delivery Strategy

## Phase 0 - Contracts and Baseline Freeze (0.5 day)

Goal:

- Freeze interfaces before coding to avoid broad refactor churn.

Tasks:

1. Define initial support scope:
   - Single-file `model.safetensors` (current `/Users/tangqi/weights` layout).
   - `float32` only for first pass.
   - Decoder-only causal LM, greedy sampling path first.
2. Define special-token source-of-truth policy:
   - priority order between `config.json`, `tokenizer_config.json`, and runtime tokenizer ids.
   - explicit conflict behavior (fail-fast vs override rule).
3. Add architecture notes in docs for key mapping rules.

Files to modify:

- docs/README or docs/new plan references
- docs/ModelExecutor.md
- docs/Scheduler.md (only if contract wording needs sync)

Acceptance:

- Team agrees on exact first-pass capability and non-goals.

---

## Phase 1 - SafeTensor Metadata + Tensor Loader Layer (2-3 days)

Goal:

- Read SafeTensor file natively in C++ and expose tensor views as `torch::Tensor`.

New files:

- include/tiny_llm/models/hf_safetensors_loader.h
- src/models/hf_safetensors_loader.cpp

Suggested APIs:

- `class HFSafeTensorLoader`
- `static HFSafeTensorLoader from_file(const std::string& path)`
- `bool has_tensor(const std::string& key) const`
- `Tensor tensor(const std::string& key) const`
- `std::vector<int64_t> shape(const std::string& key) const`
- `DType dtype(const std::string& key) const`

Implementation logic:

1. Parse SafeTensor file structure:
   - first 8 bytes: header length
   - JSON header: per-tensor dtype/shape/data_offsets
2. Validate integrity:
   - offset monotonicity
   - no overlap
   - in-range file boundaries
3. Return zero-copy tensor views where possible.
4. Map SafeTensor dtypes to existing `DType` (`kFloat32` first).

Files to modify:

- CMakeLists.txt
  - add new source file to `tiny_llm_models`
  - add JSON dependency choice (nlohmann/json or existing parser utility)

Tests to add:

- tests/test_hf_safetensors_loader.cpp

Test logic:

- can open `/Users/tangqi/weights/model.safetensors`
- can read known keys and expected shapes
- rejects malformed offset metadata

Acceptance:

- Loader returns correct tensor metadata and data pointer for at least 10 canonical Llama keys.

---

## Phase 2 - HF Config + Engine Construction Path (1-2 days)

Goal:

- Allow engine bootstrap directly from HF model directory.

Files to modify:

- include/tiny_llm/runtime/engine_args.h
- src/runtime/executor.cpp
- include/tiny_llm/models/llama_model.h

New files:

- include/tiny_llm/models/hf_llama_config_loader.h
- src/models/hf_llama_config_loader.cpp

Planned changes:

1. Extend `EngineModelType`:
   - add a value like `kHFLlamaSafeTensor`.
2. Extend `EngineArgs` with HF inputs:
   - `std::string hf_model_dir`
   - optional `std::string hf_weight_file` (default `model.safetensors`)
3. In `ModelExecutor::init_from_args`:
   - when model type is HF, call HF config loader + safetensor loader
   - build `LlamaConfig` from `config.json`
4. Expand `LlamaConfig` fields (minimum):
   - num_hidden_layers
   - hidden_size
   - intermediate_size
   - num_attention_heads
   - rms_norm_eps
   - vocab_size
   - bos/eos ids (from tokenizer/config path checks)

Special-token contract in this phase:

- If ids differ across files, emit clear error with all discovered values.
- Add one explicit normalization path and keep it deterministic.

Acceptance:

- `LLMEngine` can be constructed using only HF directory path (without manual model object wiring).

---

## Phase 3 - Weight Mapping into LLaMA Runtime (2-3 days)

Goal:

- Map HuggingFace Llama key names into internal model parameter containers.

New files:

- include/tiny_llm/models/llama_weight_map.h
- src/models/llama_weight_map.cpp

Files to modify:

- include/tiny_llm/models/llama_model.h
- src/models/llama_layer.cpp

Mapping logic (for each layer i):

- model.embed_tokens.weight
- model.layers.i.input_layernorm.weight
- model.layers.i.self_attn.q_proj.weight
- model.layers.i.self_attn.k_proj.weight
- model.layers.i.self_attn.v_proj.weight
- model.layers.i.self_attn.o_proj.weight
- model.layers.i.post_attention_layernorm.weight
- model.layers.i.mlp.gate_proj.weight
- model.layers.i.mlp.up_proj.weight
- model.layers.i.mlp.down_proj.weight
- model.norm.weight
- lm_head.weight

Design suggestion:

- Add an internal `LlamaWeights` struct holding references/tensors.
- Validate every required key exists and shape matches config.
- Fail fast with explicit error messages:
  - missing key
  - shape mismatch
  - dtype mismatch

Acceptance:

- Model initialization succeeds and all required weights are bound.

---

## Phase 4 - Real Llama Forward Path (Core Work) (4-7 days)

Goal:

- Build the real decoder-layer path based on loaded weights.

Files to modify:

- src/models/llama_layer.cpp (major rewrite)
- include/tiny_llm/models/llama_model.h
- include/tiny_llm/operators/rmsnorm.h
- src/operators/rmsnorm/rmsnorm.cpp
- include/tiny_llm/operators/matmul.h
- src/operators/matmul/matmul.cpp
- include/tiny_llm/operators/paged_attention.h
- src/operators/paged_attention/paged_attention.cpp

Likely new operator files (if needed):

- include/tiny_llm/operators/rope.h
- src/operators/rope/rope.cpp
- include/tiny_llm/operators/activation.h
- src/operators/activation/silu_swiglu.cpp

Forward logic to implement:

1. token embedding lookup
2. per-layer pre-attention RMSNorm
3. QKV projections
4. RoPE application
5. paged attention (with runtime metadata already present)
6. output projection
7. residual + post-attention RMSNorm
8. MLP (gate/up/down with SiLU-SwiGLU)
9. final norm + lm_head projection to logits

Important integration constraints:

- Keep runtime loop unchanged: schedule -> execute_model -> update_from_output
- Keep `ModelExecutor` flattened input contract unchanged.
- KV allocation remains scheduler-owned.

Acceptance:

- Single-step logits are numerically close to Python reference for the same input (small tolerance, same dtype path).

---

## Phase 5 - Llama Tokenizer Compatibility (2-3 days)

Goal:

- Ensure prompt encode/decode behavior matches HF Llama tokenizer conventions.

Files to modify:

- include/tiny_llm/runtime/tokenizer.h
- src/runtime/tokenizer.cpp
- src/runtime/processors.cpp

New files:

- include/tiny_llm/runtime/llama_tokenizer.h (optional split)
- src/runtime/llama_tokenizer.cpp (optional split)

Implementation options:

1. Preferred: load `tokenizer.model` using sentencepiece C++ runtime.
2. Alternate: parse `tokenizer.json` path if sentencepiece is not introduced.

Required behavior:

- consistent bos/eos handling with `tokenizer_config.json`
- special token ids aligned with `config.json` and model expectations
- stable incremental decode behavior in `OutPreprocessor`

For this specific model folder:

- Add a dedicated test for bos/eos/unk id reconciliation.
- Ensure generated stop behavior uses reconciled ids only.

Acceptance:

- Token IDs produced by C++ tokenizer match HF tokenizer for a fixed prompt set.

---

## Phase 6 - E2E Validation and Hardening (2-3 days)

Goal:

- Make the new path production-safe in current project scope.

Files to modify:

- tests/CMakeLists.txt
- README.md
- tools/llama_engine_generate.cpp

New tests:

- tests/test_hf_safetensors_loader.cpp
- tests/test_hf_llama_config.cpp
- tests/test_hf_llama_weight_map.cpp
- tests/test_hf_llama_runtime.cpp

Validation checklist:

1. Build passes (CPU first, then CUDA mode if enabled).
2. Existing runtime test still passes.
3. New HF loader tests pass.
4. End-to-end generation from `/Users/tangqi/weights` completes and returns valid tokens.

Acceptance:

- A dedicated example binary can run:
  - model dir: `/Users/tangqi/weights`
  - prompt input
  - streamed decode output

---

## 4. Recommended Execution Order (Issue-Level)

1. Implement Phase 1 loader and tests first.
2. Add Phase 2 engine args + constructor wiring.
3. Add Phase 3 key mapping and strict shape validation.
4. Rewrite forward path in Phase 4.
5. Add tokenizer compatibility in Phase 5.
6. Finish test matrix + docs in Phase 6.

Do not start tokenizer/full forward before loader + key map are stable.

## 5. Practical Notes for Your Current Weights Folder

Because `/Users/tangqi/weights` already has both `model.safetensors` and `pytorch_model.bin`, use this rollout strategy:

1. Primary path: `model.safetensors` (target implementation).
2. Temporary debug fallback: `pytorch_model.bin` only for parity checks while debugging loader.
3. Keep all acceptance tests pinned to safetensors path, not bin path.

## 6. Definition of Done

The task is done only when all items are true:

1. Engine can initialize from HF directory directly.
2. SafeTensor is parsed natively in C++ (not external conversion-only path).
3. Llama forward uses real HF weights.
4. Tokenizer behavior is HF-compatible for Llama.
5. New tests pass and existing runtime chain is not broken.
