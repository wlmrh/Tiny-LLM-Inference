vLLM 中使用了一个分布式框架来完成计算任务，关于如何将任务拆分，有 3 种方案：

| **变量名**     | **全称**                        | **核心含义**                                                 | **切分对象**             |
| -------------- | ------------------------------- | ------------------------------------------------------------ | ------------------------ |
| **`tp_size`**  | **Tensor Parallelism**          | **张量并行**。将模型每一层的矩阵（如线性层）纵向或横向切开。 | **层内（Intra-layer）**  |
| **`pp_size`**  | **Pipeline Parallelism**        | **流水线并行**。将模型的不同层按顺序排布在不同卡上（如 1-10 层在卡 A，11-20 层在卡 B）。 | **层间（Inter-layer）**  |
| **`pcp_size`** | **Prefill Context Parallelism** | **预填充上下文并行**。将超长的 Prompt 序列切成几段，让多张卡同时算不同的 Token 段。 | **序列维度（Sequence）** |

在绝大多数情况下，每个 GPU 上都运行着一个 Worker 进程，对应一个 rank。此时，本 node 上的 Worker 数量为 `local_world_size = tp_size * pp_size * pcp_size`。

Worker 与 Executor 运行在不同进程中，通信的细节被封装在 `rpc_broadcast_mq: MessageQueue` 中，通过 zmq / shared memory 将数据发送到 WorkerProc 进程的 `rpc_broadcast_mq: MessageQueue` 中。每个 `Worker` 被创建时，都通过句柄连接到全局唯一的 `rpc_broadcast_mq`（而不是创建新的队列，全局只有一个），从而可以接收到 `MultiprocExecutor` 发来的消息；同样地，每个 `Worker` 也会通过句柄连接到专属的 `worker_response_mq`（全局共有 N 个），用于返回 `Worker` 推理的结果。
