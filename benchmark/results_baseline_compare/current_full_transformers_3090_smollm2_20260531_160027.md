# Industrial Benchmark Report: current_full_transformers_3090_smollm2

- generated_at: `2026-05-31T16:00:27+08:00`
- model: `/models/smollm2-135M`
- device: `cuda:0`
- gpu: `NVIDIA GeForce RTX 3090`, memory `49152 MiB`
- warmup/repeat: `1/3`
- profile_detail: `off`

Current benchmark mode is offline batched generation. It is not a request-rate server benchmark with p50/p95/p99 latency.

| scenario | backend | batch | ISL target | prompt tokens | OSL | generated | TTFT ms | latency ms | decode ms/token | e2e tok/s | decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interactive | tinyllm | 1 | 128 | 128 | 64 | 64.000 | 18.396 | 464.158 | 6.804 | 137.884 | 141.331 |
| interactive | transformers | 1 | 128 | 128 | 64 | 64.000 | 32.296 | 1666.279 | 25.936 | 38.409 | 38.556 |
| chat_serving | tinyllm | 8 | 128 | 1024 | 128 | 1024.000 | 31.250 | 1342.620 | 1.088 | 762.688 | 774.762 |
| chat_serving | transformers | 8 | 128 | 1024 | 128 | 1024.000 | 36.578 | 3541.924 | 3.450 | 289.108 | 289.843 |
| long_prefill | tinyllm | 4 | 1024 | 4095 | 64 | 256.000 | 468.918 | 3606.179 | 6.393 | 70.989 | 80.325 |
| long_prefill | transformers | 4 | 1024 | 4095 | 64 | 256.000 | 111.285 | 1789.269 | 6.659 | 143.075 | 150.180 |
| decode_heavy | tinyllm | 4 | 256 | 1024 | 256 | 1024.000 | 41.530 | 3910.120 | 3.491 | 261.885 | 263.662 |
| decode_heavy | transformers | 4 | 256 | 1024 | 256 | 1024.000 | 38.896 | 6897.299 | 6.724 | 148.464 | 148.723 |
| throughput | tinyllm | 16 | 128 | 2048 | 128 | 2048.000 | 32.712 | 1718.127 | 0.628 | 1191.996 | 1205.638 |
| throughput | transformers | 16 | 128 | 2048 | 128 | 2048.000 | 46.805 | 3520.689 | 1.710 | 581.704 | 584.936 |

## TinyLLM / Transformers Ratios

| scenario | latency | TTFT | e2e throughput | decode throughput | load/init |
| --- | ---: | ---: | ---: | ---: | ---: |
| interactive | 0.279 | 0.570 | 3.590 | 3.666 | 0.952 |
| chat_serving | 0.379 | 0.854 | 2.638 | 2.673 | 0.792 |
| long_prefill | 2.015 | 4.214 | 0.496 | 0.535 | 0.843 |
| decode_heavy | 0.567 | 1.068 | 1.764 | 1.773 | 1.089 |
| throughput | 0.488 | 0.699 | 2.049 | 2.061 | 0.801 |

## Prompts And Outputs

### interactive

#### Prompt 0 (128 tokens)
```
Request 0: You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、
```

##### tinyllm output (64 tokens, finish=length)
```
处理缓存、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、
```

##### transformers output (64 tokens, finish=length)
```
处理缓存、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、
```

### chat_serving

#### Prompt 0 (128 tokens)
```
Request 0: You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、
```

##### tinyllm output (128 tokens, finish=length)
```
处理缓存、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速
```

##### transformers output (128 tokens, finish=length)
```
处理缓存、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速
```

#### Prompt 1 (128 tokens)
```
Request 1: 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, determ
```

##### tinyllm output (128 tokens, finish=length)
```
inate response times, and a consistent response time.

The benchmark should reflect the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world.
```

##### transformers output (128 tokens, finish=length)
```
inate response times, and a consistent response time.

The benchmark should reflect the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world.
```

#### Prompt 2 (128 tokens)
```
Request 2: The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LL
```

##### tinyllm output (128 tokens, finish=length)
```
-based benchmark that uses a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-
```

##### transformers output (128 tokens, finish=length)
```
-based benchmark that uses a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-
```

#### Prompt 3 (128 tokens)
```
Request 3: 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with conc
```

##### tinyllm output (128 tokens, finish=length)
```
at.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA
```

##### transformers output (128 tokens, finish=length)
```
at.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA
```

#### Prompt 4 (128 tokens)
```
Request 4: You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、
```

##### tinyllm output (128 tokens, finish=length)
```
处理缓存、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速
```

##### transformers output (128 tokens, finish=length)
```
处理缓存、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速
```

#### Prompt 5 (128 tokens)
```
Request 5: 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, determ
```

##### tinyllm output (128 tokens, finish=length)
```
inate response times, and a consistent response time.

The benchmark should also reflect the performance of the system. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number
```

##### transformers output (128 tokens, finish=length)
```
inate response times, and a consistent response time.

The benchmark should also reflect the performance of the system. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number
```

#### Prompt 6 (128 tokens)
```
Request 6: The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LL
```

##### tinyllm output (128 tokens, finish=length)
```
-based benchmark that uses a single token per request. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior.

The benchmark should be able to measure the following:

  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
```

##### transformers output (128 tokens, finish=length)
```
-based benchmark that uses a single token per request. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior.

The benchmark should be able to measure the following:

  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
```

#### Prompt 7 (128 tokens)
```
Request 7: 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with conc
```

##### tinyllm output (128 tokens, finish=length)
```
at.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA
```

##### transformers output (128 tokens, finish=length)
```
at.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA
```

### long_prefill

#### Prompt 0 (1024 tokens)
```
Request 0: You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入
```

##### tinyllm output (64 tokens, finish=length)
```
输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged
```

##### transformers output (64 tokens, finish=length)
```
输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged
```

#### Prompt 1 (1023 tokens)
```
Request 1: 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 
```

##### tinyllm output (64 tokens, finish=length)
```
任意预填充。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工�
```

##### transformers output (64 tokens, finish=length)
```
任意预填充。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工�
```

#### Prompt 2 (1024 tokens)
```
Request 2: The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存
```

##### tinyllm output (64 tokens, finish=length)
```
、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched
```

##### transformers output (64 tokens, finish=length)
```
、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched
```

#### Prompt 3 (1024 tokens)
```
Request 3: 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加
```

##### tinyllm output (64 tokens, finish=length)
```
载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，�
```

##### transformers output (64 tokens, finish=length)
```
载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，�
```

### decode_heavy

#### Prompt 0 (256 tokens)
```
Request 0: You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、rep
```

##### tinyllm output (256 tokens, finish=length)
```
etitive measurements、可见性、线上稳定性。

## 2.2.2.3.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.
```

##### transformers output (256 tokens, finish=length)
```
etitive measurements、可见性、线上稳定性。

## 2.2.2.3.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.
```

#### Prompt 1 (256 tokens)
```
Request 1: 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput,
```

##### tinyllm output (256 tokens, finish=length)
```
 and the impact of the model on the system. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. The
```

##### transformers output (256 tokens, finish=length)
```
 and the impact of the model on the system. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. The
```

#### Prompt 2 (256 tokens)
```
Request 2: The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模
```

##### tinyllm output (256 tokens, finish=length)
```
型缓存、高速缓存、高速预填充、高速缓存、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞
```

##### transformers output (256 tokens, finish=length)
```
型缓存、高速缓存、高速预填充、高速缓存、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞吐、高速吞
```

#### Prompt 3 (256 tokens)
```
Request 3: 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy
```

##### tinyllm output (256 tokens, finish=length)
```
 scheduling, and a high-quality, high-throughput, and high-performance LLM inference engine. 请用工程评审的方式分析一个大语言模型推理系统，提供一些真正的硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode throughput、tokenizer compatibility、CUDA execution、和线上稳定性。

## 2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
```

##### transformers output (256 tokens, finish=length)
```
 scheduling, and a high-quality, high-throughput, and high-performance LLM inference engine. 请用工程评审的方式分析一个大语言模型推理系统，提供一些真正的硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode throughput、tokenizer compatibility、CUDA execution、和线上稳定性。

## 2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1
```

### throughput

#### Prompt 0 (128 tokens)
```
Request 0: You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、
```

##### tinyllm output (128 tokens, finish=length)
```
处理缓存、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速
```

##### transformers output (128 tokens, finish=length)
```
处理缓存、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速
```

#### Prompt 1 (128 tokens)
```
Request 1: 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, determ
```

##### tinyllm output (128 tokens, finish=length)
```
inate response times, and a consistent response time.

The benchmark should reflect the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world.
```

##### transformers output (128 tokens, finish=length)
```
inate response times, and a consistent response time.

The benchmark should reflect the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world. The benchmark should be able to measure the performance of the system in the real world.
```

#### Prompt 2 (128 tokens)
```
Request 2: The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LL
```

##### tinyllm output (128 tokens, finish=length)
```
-based benchmark that uses a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-
```

##### transformers output (128 tokens, finish=length)
```
-based benchmark that uses a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-shot, deterministic, greedy decoding algorithm. The benchmark is designed to measure the performance of a single-
```

#### Prompt 3 (128 tokens)
```
Request 3: 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with conc
```

##### tinyllm output (128 tokens, finish=length)
```
at.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA
```

##### transformers output (128 tokens, finish=length)
```
at.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA
```

#### Prompt 4 (128 tokens)
```
Request 4: You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、
```

##### tinyllm output (128 tokens, finish=length)
```
处理缓存、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速
```

##### transformers output (128 tokens, finish=length)
```
处理缓存、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速
```

#### Prompt 5 (128 tokens)
```
Request 5: 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, determ
```

##### tinyllm output (128 tokens, finish=length)
```
inate response times, and a consistent response time.

The benchmark should also reflect the performance of the system. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number
```

##### transformers output (128 tokens, finish=length)
```
inate response times, and a consistent response time.

The benchmark should also reflect the performance of the system. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number of requests in a reasonable amount of time. The system should be able to handle a large number
```

#### Prompt 6 (128 tokens)
```
Request 6: The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LL
```

##### tinyllm output (128 tokens, finish=length)
```
-based benchmark that uses a single token per request. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior.

The benchmark should be able to measure the following:

  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
```

##### transformers output (128 tokens, finish=length)
```
-based benchmark that uses a single token per request. The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior.

The benchmark should be able to measure the following:

  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
  • The number of tokens generated per request.
```

#### Prompt 7 (128 tokens)
```
Request 7: 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with conc
```

##### tinyllm output (128 tokens, finish=length)
```
at.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA
```

##### transformers output (128 tokens, finish=length)
```
at.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concat.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA
```

#### Prompt 8 (128 tokens)
```
Request 8: You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、
```

##### tinyllm output (128 tokens, finish=length)
```
处理缓存、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速
```

##### transformers output (128 tokens, finish=length)
```
处理缓存、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速
```

#### Prompt 9 (128 tokens)
```
Request 9: 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, determ
```

##### tinyllm output (128 tokens, finish=length)
```
inate response times, and a consistent response time.

The benchmark should also reflect the performance of the server and the server's response time. The server should be able to handle a large number of requests in a reasonable amount of time. The server should be able to handle a large number of requests in a reasonable amount of time. The server should be able to handle a large number of requests in a reasonable amount of time. The server should be able to handle a large number of requests in a reasonable amount of time. The server should be able to handle a large number of requests in a reasonable amount of time. The server should be
```

##### transformers output (128 tokens, finish=length)
```
inate response times, and a consistent response time.

The benchmark should also reflect the performance of the server and the server's response time. The server should be able to handle a large number of requests in a reasonable amount of time. The server should be able to handle a large number of requests in a reasonable amount of time. The server should be able to handle a large number of requests in a reasonable amount of time. The server should be able to handle a large number of requests in a reasonable amount of time. The server should be able to handle a large number of requests in a reasonable amount of time. The server should be
```

#### Prompt 10 (128 tokens)
```
Request 10: The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an off
```

##### tinyllm output (128 tokens, finish=length)
```
-the-shelf benchmark that is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure
```

##### transformers output (128 tokens, finish=length)
```
-the-shelf benchmark that is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure
```

#### Prompt 11 (128 tokens)
```
Request 11: 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling w
```

##### tinyllm output (128 tokens, finish=length)
```
/o loss.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling w/o loss.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling w/o loss.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer
```

##### transformers output (128 tokens, finish=length)
```
/o loss.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling w/o loss.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling w/o loss.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer
```

#### Prompt 12 (128 tokens)
```
Request 12: You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐
```

##### tinyllm output (128 tokens, finish=length)
```
、处理缓存、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高�
```

##### transformers output (128 tokens, finish=length)
```
、处理缓存、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高速缓存、高速预填充、高�
```

#### Prompt 13 (128 tokens)
```
Request 13: 请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests,
```

##### tinyllm output (128 tokens, finish=length)
```
 and a variety of other features.

The benchmark should also reflect the performance of the system. The benchmark should be able to measure the performance of the system in real time. The benchmark should also be able to measure the performance of the system in real time. The benchmark should also be able to measure the performance of the system in real time. The benchmark should also be able to measure the performance of the system in real time. The benchmark should also be able to measure the performance of the system in real time. The benchmark should also be able to measure the performance of the system in real time. The benchmark should also be able to
```

##### transformers output (128 tokens, finish=length)
```
 and a variety of other features.

The benchmark should also reflect the performance of the system. The benchmark should be able to measure the performance of the system in real time. The benchmark should also be able to measure the performance of the system in real time. The benchmark should also be able to measure the performance of the system in real time. The benchmark should also be able to measure the performance of the system in real time. The benchmark should also be able to measure the performance of the system in real time. The benchmark should also be able to measure the performance of the system in real time. The benchmark should also be able to
```

#### Prompt 14 (128 tokens)
```
Request 14: The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an off
```

##### tinyllm output (128 tokens, finish=length)
```
-the-shelf benchmark that is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure
```

##### transformers output (128 tokens, finish=length)
```
-the-shelf benchmark that is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure the performance of a particular application. The benchmark is designed to measure
```

#### Prompt 15 (128 tokens)
```
Request 15: 在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling w
```

##### tinyllm output (128 tokens, finish=length)
```
/o loss.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling w/o loss.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling w/o loss.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer
```

##### transformers output (128 tokens, finish=length)
```
/o loss.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling w/o loss.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling w/o loss.

You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer
```

