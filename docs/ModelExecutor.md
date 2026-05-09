着重关注 `ModelExecutor`中的 `execute_model` 函数，该函数接收来自 Scheduler 的调度结果 `scheduler_output`。（每个被调度的请求以及完成该请求需要的必要步骤）将它们整合在一起交给模型层进行计算。

需要整合的信息包括：

```
input_values: 输入 token 序列
position_values: 每个 token 在其 Sequence 中的位置，用于辅助计算 RoPE 或者位置旋转 Q/K
slot_values: 每个输入 token 计算出来的 kv 应该存放在 KV Cache 的哪一个位置
seq_index_values: 
context_values: 
block_table_values: 
```

