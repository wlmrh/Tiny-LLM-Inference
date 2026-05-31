# Models and Weights Module

This module implements HuggingFace LLaMA-style model loading and inference. Qwen2-family checkpoints are supported through the same LLaMA-style runtime path when their architecture matches the implemented layer layout.

## Main Files

- `include/tiny_llm/models/model.h`
- `include/tiny_llm/models/llama_config.h`
- `include/tiny_llm/models/hf_llama_config_loader.h`
- `src/models/hf_llama_config_loader.cpp`
- `include/tiny_llm/models/hf_safetensors_loader.h`
- `src/models/hf_safetensors_loader.cpp`
- `include/tiny_llm/models/llama_weight_map.h`
- `src/models/llama_weight_map.cpp`
- `include/tiny_llm/models/llama_model.h`
- `src/models/llama_model.cpp`
- `include/tiny_llm/models/llama_decoder_layer.h`
- `src/models/llama_layer.cpp`
- `include/tiny_llm/models/modules/*.h`
- `src/models/modules/*.cpp`

## `Model`

`Model` is the runtime model contract used by `ModelRunner`.

Interfaces:

- `num_layers()`: transformer layer count.
- `vocab_size()`: logits vocabulary size.
- `forward(const PreparedInputs&, RuntimeContext&)`: computes logits for flattened scheduler input.
- `expected_bos_id()`, `expected_eos_id()`, `expected_unk_id()`: optional tokenizer contract IDs.

## `LlamaConfig`

`LlamaConfig` mirrors HuggingFace `config.json` fields:

- vocabulary and hidden dimensions;
- layer/head counts;
- GQA key/value head count;
- head dimension;
- max positions;
- BOS/EOS/UNK/PAD IDs;
- RMSNorm epsilon;
- RoPE theta and scaling fields;
- activation name;
- model type and dtype metadata.

`HFLlamaConfigLoader` validates required dimensions, divisibility rules, special token IDs, optional JSON null fields, and LLaMA 3 RoPE scaling metadata.

## Safetensors and Weight Maps

`HFSafeTensorInfo` records:

- shape;
- runtime dtype;
- storage dtype string;
- byte offset;
- byte size.

`HFSafeTensorLoader` interfaces:

- `from_file(path)`
- `has_tensor(key)`
- `tensor(key)`
- `shape(key)`
- `dtype(key)`
- `keys()`
- `file_path()`

`WeightMap` is a name-to-tensor registry. It keeps tensor handles alive while exposing raw pointers for low-level binding.

Interfaces:

- `from_safetensors(loader)`
- `from_safetensors(loader, parallel_config)`
- `add_tensor(name, tensor)`
- `add_tensor(name, data, shape, dtype)`
- `contains(name)`
- `get_tensor(name)`
- `get_tensor_view(name)`
- `get_tensor_as<T>(name)`
- `keys()`

## `LlamaForCausalLM`

`LlamaForCausalLM` implements the `Model` interface.

Attributes:

- `config_`: model configuration.
- `model_`: owned `LlamaModel`.
- `lm_head_`: output projection. If `lm_head.weight` is absent, it uses tied `model.embed_tokens.weight`.

Interfaces:

- `allocate_buffers(max_batch_size)`
- `allocate_buffers(max_batch_size, parallel_config)`
- `num_layers()`
- `vocab_size()`
- `expected_bos_id()`, `expected_eos_id()`, `expected_unk_id()`
- `forward(inputs, ctx)`
- `compute_logits(hidden_states, ctx)`

## `LlamaModel`

`LlamaModel` contains embeddings, decoder layers, and final norm.

Attributes:

- `config_`
- `layers_`
- `embed_tokens_`
- `final_norm_`
- `allocated_max_batch_size_`
- `buffer_parallel_config_`
- `buffers_`: reusable owned tensors for model/layer intermediate states.

Interfaces:

- `allocate_buffers(max_batch_size, parallel_config)`
- `forward_hidden(inputs, ctx)`
- `num_layers()`, `vocab_size()`, `hidden_size()`, `config()`

Buffers are allocated once per device/max batch size and then sliced into batch views for each forward pass.

## Decoder Layer Components

`LlamaDecoderLayer` attributes:

- `input_layernorm_`
- `self_attn_`
- `post_attention_layernorm_`
- `mlp_`
- `layer_id_`
- `config_`

Interfaces:

- `load_weights(weight_map, layer_id)`
- `forward(hidden_states, positions, buffers, ctx)`

`LlamaSelfAttention` attributes:

- stacked Q/K/V projection descriptors;
- `qkv_proj_`;
- `o_proj_`;
- `rotary_`;
- `layer_id_`.

It runs QKV projection, splits Q/K/V, applies RoPE, calls paged attention, and applies output projection.

`LlamaMLP` attributes:

- stacked gate/up projection descriptors;
- `gate_up_proj_`;
- `down_proj_`.

It supports `hidden_act == "silu"` and computes `down_proj(silu(gate) * up)`.

## Reusable Model Modules

`Embedding`:

- supports `[vocab, hidden]` and `[hidden, vocab]` layouts;
- binds a weight tensor;
- performs token embedding lookup.

`Linear`:

- supports `kInOut` and `kOutIn` weight layouts;
- supports single or stacked weights;
- caches stacked weights/biases for combined matmul when appropriate.

`RMSNorm`:

- binds scale weights;
- calls `ops::rmsnorm`.

`RotaryEmbedding`:

- stores RoPE shape and scaling parameters;
- caches inverse frequency per device;
- can use cached cos/sin values;
- supports LLaMA 3 RoPE scaling.
