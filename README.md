# Tiny-LLM-Inference

Tiny-LLM-Inference is a compact C++17 inference engine inspired by vLLM. It is designed for learning, debugging, and validating the core mechanics of modern decoder-only LLM serving without hiding the interesting parts behind a large framework.

The current implementation focuses on a single-process runtime for small LLaMA-family models, with SmolLM2-135M as the main Hugging Face reference model.

## Highlights

- Frontend/core runtime split:
  - `LLMEngine` owns text/tokenizer-facing request handling.
  - `EngineCore` runs scheduling, model execution, and output updates over token IDs.
- FCFS continuous batching with chunked prefill, one-token decode steps, and simplified tail preemption.
- Hugging Face LLaMA/SmolLM2 loading from `config.json`, `model.safetensors`, and `tokenizer.json`.
- Standard LLaMA building blocks:
  - RoPE
  - RMSNorm
  - SwiGLU MLP
  - grouped-query attention (GQA)
  - tied or untied LM heads, depending on the model config
- CPU and CUDA runtime KV cache with paged metadata and physical KV block storage.
- Tensor alignment tools for comparing C++ intermediate tensors, logits, and greedy generation against PyTorch/Transformers.
- Optional CUDA build path with single-GPU SmolLM2 generation smoke coverage. The current CUDA paged-attention path is a correctness bridge, not the final optimized kernel.

## Repository Layout

```text
include/tiny_llm/        Public C++ headers
src/                     Engine, runtime, operators, models, and backends
examples/                Small runnable examples
tools/                   Standalone C++ debugging utilities
scripts/                 Python comparison and inspection scripts
tests/                   CTest entry points
docs/                    Design notes and implementation plans
assets/tiny_lm/          Tiny toy model assets
```

Debug utilities intentionally live under `tools/`, not `tests/`. The `tests/` tree is reserved for automated test entry points.

## Dependencies

Required:

- CMake 3.18 or newer
- A C++17 compiler
- Python 3, recommended for PyTorch discovery and comparison scripts
- PyTorch/libtorch
- Rust `cargo`, required by `tokenizers-cpp`

Optional:

- CUDA Toolkit and cuBLAS when building with `TINYLLM_ENABLE_CUDA=ON`
- Python packages for reference comparisons:
  - `torch`
  - `transformers`
  - `safetensors`

CMake first tries `find_package(Torch)`. If Torch is not found, it tries Python's `torch.utils.cmake_prefix_path`, then `$HOME/libtorch*`.

## Build

CPU build:

```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=OFF
cmake --build build -j
```

CUDA build:

```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=ON
cmake --build build -j
```

CUDA build with Python PyTorch/libtorch and an explicit CUDA toolkit:

```bash
cmake -S . -B build-cuda \
  -DTINYLLM_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-cuda -j
```

Manually point CMake at libtorch:

```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=OFF -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build -j
```

Use Python PyTorch as the libtorch provider:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j
```

## Model Setup

The LLaMA-family debugging and validation flow expects a local Hugging Face model directory containing at least:

```text
config.json
model.safetensors
tokenizer.json
```

For the main validated setup, place SmolLM2-135M at:

```bash
~/models/smollm2-135M
```

or pass the model directory explicitly to the tools/scripts.

## Run Examples

Toy TinyEmbeddingLM example:

```bash
./build/tiny_lm_inference
```

MiniLLaMA example:

```bash
./build/llama_inference
```

Run Hugging Face LLaMA/SmolLM2 greedy generation through the engine:

```bash
./build/tools/llama_engine_generate \
  ~/models/smollm2-135M \
  8 \
  hello \
  "tiny llm inference"
```

The tool prints one JSON object per prompt, including generated token IDs and decoded output text.

Run the same tool on CUDA:

```bash
./build-cuda/tools/llama_engine_generate \
  --device cuda:0 \
  ~/models/smollm2-135M \
  8 \
  hello \
  "tiny llm inference"
```

## Test

Run all registered CTest tests:

```bash
ctest --test-dir build --output-on-failure
```

Run the runtime smoke test with a local SmolLM2 model:

```bash
TINYLLM_HF_TINY_LLAMA_DIR=~/models/smollm2-135M \
ctest --test-dir build --output-on-failure
```

The Transformers comparison tests return CTest skip code `77` when their model path or Python dependencies are unavailable.

CUDA CTest with SmolLM2 generation smoke:

```bash
TINYLLM_HF_TINY_LLAMA_DIR=~/models/smollm2-135M \
LD_LIBRARY_PATH=/path/to/torch/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
ctest --test-dir build-cuda --output-on-failure
```

When `TINYLLM_ENABLE_CUDA=ON`, CTest registers `test_llama_generation_cuda_smoke`, which runs `llama_engine_generate --device cuda:0` and checks deterministic SmolLM2 token IDs for `hello` and `tiny llm inference`.

## Compare Against Transformers

The project includes tools and scripts for tensor alignment against PyTorch/Transformers. These are the preferred diagnostics when changing model loading, operators, attention, or scheduling.

Compare safetensors loading:

```bash
python3 scripts/compare_hf_safetensor_with_pytorch.py \
  --dump-binary build/tools/hf_safetensor_dump \
  --model-dir ~/models/smollm2-135M
```

Compare logits:

```bash
TINYLLM_HF_TINY_LLAMA_DIR=~/models/smollm2-135M \
python3 scripts/compare_llama_logits_with_transformers.py \
  --dump-binary build/tools/llama_logits_dump
```

Compare intermediate tensors:

```bash
python3 scripts/compare_llama_tensors_with_transformers.py \
  --dump-binary build/tools/llama_tensor_dump \
  --model-dir ~/models/smollm2-135M \
  --tokens 1 22172 318
```

Compare full greedy generation:

```bash
python3 scripts/compare_llama_generation_with_transformers.py \
  --engine-binary build/tools/llama_engine_generate \
  --model-dir ~/models/smollm2-135M \
  --max-new-tokens 8 \
  --prompt hello \
  --prompt "tiny llm inference"
```

Expected output should report `MATCH` for generated token IDs and decoded text.

## Runtime Architecture

At a high level, a generation step follows this path:

```text
LLMEngine
  -> EngineCore::step()
    -> Scheduler::schedule()
    -> ModelExecutor::execute_model()
    -> Model::forward_step()
    -> Scheduler::update_from_output()
```

The scheduler owns request state and the runtime KV cache. `EngineCore` passes the same scheduler-owned `KVCache*` to `ModelExecutor`, so attention reads and writes the physical blocks allocated by scheduling decisions.

Paged attention metadata uses:

```text
slot_mapping[B]
seq_indices[B]
context_lens[num_seqs]
block_tables[num_layers, num_seqs, max_blocks_per_seq]
```

For CPU LLaMA runtime, each physical KV block stores float32 data as:

```text
[K block][V block]
```

where each side contains `block_size_tokens * (num_key_value_heads * head_dim)` floats.

`llama_tensor_dump` and direct `LlamaModel` alignment tools can still run without runtime KV metadata. In that mode, LLaMA attention uses an in-batch causal fallback so full-sequence tensor alignment remains straightforward.

On CUDA, `llama_attention` currently uses a torch-based paged-attention bridge:

- runtime tensors, model weights, workspace, and KV blocks are placed according to `EngineArgs::parallel_config`
- metadata tensors may live on CUDA, but validation/control metadata is copied to CPU before indexing
- KV block storage is wrapped with `torch::from_blob(... device=kv_cache.device())`
- current-step K/V are written into paged KV blocks, then gathered by `block_tables`, `seq_indices`, and positions
- attention math uses libtorch tensor ops (`matmul`, `softmax`, `matmul`) for correctness

This path is intentionally a correctness bridge for single-GPU execution. It should be replaced by a dedicated paged-attention CUDA kernel before any performance claim.

## Current Scope and Limitations

- CPU float32 inference and single-GPU CUDA SmolLM2 greedy generation are validated.
- CUDA paged attention is currently torch-based and correctness-oriented, not fused or optimized.
- The engine is single-process and single-device; multi-GPU/tensor parallel/pipeline parallel are not implemented.
- Sampling is intentionally minimal; greedy generation is the main reference path.
- The scheduler is intentionally small and FCFS-oriented.
- The project is a learning and validation engine, not a production serving system.

## Troubleshooting

### Torch is not found

If CMake reports:

```text
Torch not found. Set Torch_DIR/CMAKE_PREFIX_PATH to libtorch, or install Python torch.
```

Install Python PyTorch or pass `CMAKE_PREFIX_PATH`/`Torch_DIR` explicitly:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch
```

### Cargo is not found

`tokenizers-cpp` requires Rust `cargo`.

Install Rust from:

```text
https://rustup.rs/
```

### tokenizers-cpp configure tries to access GitHub repeatedly

`tokenizers-cpp` is fetched through CMake `FetchContent`. After the dependency has been populated once, the project keeps `FETCHCONTENT_UPDATES_DISCONNECTED` enabled so repeated configures do not depend on GitHub availability. If a build tree was created before this setting existed, rerun CMake after updating the repository, or create a fresh build directory.

### Transformers comparisons are skipped

The comparison scripts skip with code `77` when one of these is missing:

- `TINYLLM_HF_TINY_LLAMA_DIR` or `--model-dir`
- `torch`
- `transformers`
- the compiled C++ tool binary

Set the model path explicitly to make the checks run:

```bash
export TINYLLM_HF_TINY_LLAMA_DIR=~/models/smollm2-135M
```

### macOS libtorch/OpenMP issues

Some libtorch distributions depend on `libomp.dylib`. If macOS cannot locate it at runtime, install OpenMP with your package manager and ensure the dynamic library search path or install name matches your local setup.

## Development Notes

- Prefer public headers under `include/tiny_llm/...` for new code.
- Keep debug executables in `tools/` and Python diagnostics in `scripts/`.
- Use tensor/logit/generation comparison scripts after changes to model loading, operators, attention, scheduler metadata, or KV-cache behavior.
- Avoid restoring historical `test_llama_phase*` bring-up files as regular tests; ongoing LLaMA coverage should use focused smoke tests and Transformers alignment scripts.
