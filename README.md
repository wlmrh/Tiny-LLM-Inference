# Tiny-LLM-Inference

## Project Overview

Tiny-LLM-Inference is a compact C++17, single-process decoder-only LLM inference engine inspired by vLLM. It is designed for offline generation with request scheduling, paged KV cache management, Hugging Face tokenizer/safetensors loading, model execution, and configurable sampling.

The runtime targets LLaMA-style checkpoints, including small LLaMA/SmolLM2-compatible models and Qwen2-family models such as Qwen2.5-1.5B-Instruct.

## Status and Scope

- Single-process offline inference runtime.
- Local Hugging Face checkpoint directories only; no model download layer is provided.
- CPU is the default backend; CUDA supports one selected device when enabled at build time.
- No HTTP/gRPC server is included. Use the C++ API example or CLI generation tools for local inference.

## Features

- High-level `LLM` facade for batch generation over prompt strings.
- `LLMEngine` / `EngineCore` split between text/token I/O and token-level scheduling/execution.
- Scheduler-managed waiting/running queues, chunked prefill/decode, preemption, and paged KV cache ownership.
- Hugging Face `tokenizer.json` or `tokenizer.model`, single-file `model.safetensors`, and sorted sharded safetensors.
- LLaMA/SmolLM2/Qwen2-family model path with CPU execution and optional CUDA kernels.
- Default greedy decoding, HuggingFace-style repetition penalty, and seeded temperature/top-k/top-p sampling.

## Quickstart

Use the high-level C++ API when you want a vLLM-style offline generation entry point:

```cpp
#include "tiny_llm/runtime/llm.h"

#include <iostream>
#include <vector>

int main()
{
    tiny_llm::LLM llm("/models/smollm2-135M");

    tiny_llm::LLMSamplingParams params;
    params.temperature = 0.8f;
    params.top_p = 0.95f;

    std::vector<std::string> prompts = {
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    };

    const auto outputs = llm.generate(prompts, params);
    for (const auto& output : outputs)
    {
        std::cout << "Prompt: " << output.prompt << "\n";
        std::cout << "Output: " << output.text << "\n";
    }
}
```

The same flow is available through the built example:

```bash
./build/offline_llm /models/smollm2-135M cpu
./build-cuda/offline_llm /models/Qwen2.5-1.5B-Instruct cuda:0
```

## Requirements

- CMake 3.18+
- C++17 compiler
- Python with PyTorch/libtorch available to CMake
- Rust `cargo` for `tokenizers-cpp`
- CUDA Toolkit when building with `TINYLLM_ENABLE_CUDA=ON`

## Build

CPU build:

```bash
cmake -S . -B build -DTINYLLM_ENABLE_CUDA=OFF
cmake --build build -j
```

CUDA build:

```bash
cmake -S . -B build-cuda \
  -DTINYLLM_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-cuda -j
```

Main outputs are the `tiny_llm` static library, the `offline_llm` API example, generation/debug tools under `build/tools/`, and tests under `build/tests/`.

## Run Generation

Use a local Hugging Face checkpoint directory containing `config.json`, `tokenizer.json` or `tokenizer.model`, and safetensors weights.

For direct JSONL output or explicit KV block sizing, use `llama_engine_generate`:

```bash
./build/tools/llama_engine_generate /models/smollm2-135M 8 hello
```

```bash
./build-cuda/tools/llama_engine_generate \
  --device cuda:0 \
  /models/Qwen2.5-1.5B-Instruct \
  8 \
  hello
```

`llama_engine_generate` prints one JSON object per prompt and also supports `--kv-num-blocks N`.

## Benchmarks

The preferred reproducible benchmark entrypoint is the config-driven suite:

```bash
python3 benchmark/run_benchmark_suite.py \
  --config benchmark/configs/qwen25_quick.json \
  --backend tinyllm,transformers
```

It writes workload JSONL, TinyLLM request event traces, summary JSON, and Markdown reports under
`benchmark/results/`. The lower-level `llama_engine_benchmark` binary also supports
`--workload-jsonl`, full sampling flags, and `--events-jsonl` for request-level timing.

## Tests

Run the default test suite after a CPU build:

```bash
ctest --test-dir build --output-on-failure
```

## Documentation

Start with [docs/README.md](docs/README.md) for architecture and module-level documentation. Repository-specific agent workflow and coding conventions are in [AGENTS.md](AGENTS.md).
