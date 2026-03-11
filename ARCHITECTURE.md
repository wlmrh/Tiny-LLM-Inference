对于一个旨在实现“从零构建”并关注性能优化的 C++ 推理引擎，良好的工程结构不仅能提高开发效率，还能让你在后续进行内核替换（比如用自定义算子替换 cuBLAS）时更加游刃有余。

基于你 `Design.md` 中提出的六层架构设计，建议采用以下工程目录规划：

## 📂 项目目录结构规划

```text
Tiny-LLM-Inference/
├── cmake/                  # 存放自定义 CMake 脚本（如查找 CUDA 或设置编译选项）
├── include/                # 公共头文件 (.h / .hpp)
│   ├── core/               # Tensor 抽象与内存管理 (Layer 4)
│   ├── operators/          # 算子接口定义 (Layer 3)
│   ├── models/             # 模型组装逻辑 (Layer 2)
│   └── utils/              # 宏定义、CUDA 错误检查、日志
├── src/                    # 源文件实现 (.cpp)
│   ├── core/               # Tensor 与 Allocator 实现
│   ├── operators/          # 算子分发逻辑
│   └── models/             # LLaMA 具体的层实现
├── kernels/                # 原始 CUDA 内核 implemented in .cu (Layer 5)
│   ├── gemm/               # 不同版本的 GEMM 内核
│   ├── attention/          # PagedAttention 相关内核
│   └── activation/         # SiLU, RMSNorm 等逐元素内核
├── scripts/                # Python 脚本（用于权重转换、对比验证）
├── tests/                  # 单元测试 (GTest)
├── third_party/            # 第三方库（如 GoogleTest, Optional: cuBLAS）
├── weights/                # 转换后的二进制权重（不建议提交到 Git）
├── CMakeLists.txt          # 主编译配置
└── main.cpp                # 应用程序入口 (Layer 1)

```

---

## 🛠 关键设计说明

### 1. 头文件与实现的分离 (Header-Only vs. Compiled)

* **`include/` vs `src/**`: 将 API 定义放在 `include/` 下，实现放在 `src/`。这能确保你的算子逻辑对应用层透明，同时也方便后续将项目打包成库。
* **CUDA 内核隔离**: 建议将 `.cu` 文件独立放在 `kernels/` 目录下。在 `src/operators/` 中调用这些内核。这样做的好处是你可以轻松对比“纯 C++ 实现”与“CUDA 实现”。

### 2. 现代 CMake 构建管理

为了保持项目的可维护性，建议在主 `CMakeLists.txt` 中使用 **Target-based CMake**。

* 定义一个核心 Library 目标（例如 `tiny_llm_core`），它包含 Tensor 和内存管理。
* 定义算子目标，并链接 CUDA 库。
* **内存优化开关**: 可以通过 CMake 选项控制是否开启特定优化（如 `-DENABLE_FUSION=ON`）。

### 3. 权重管理与转换

由于 TinyLLaMA-1.1B 的权重较大，直接在 C++ 中解析 PyTorch 的 `.pth` 或 `.safetensors` 非常繁琐。

* **工程建议**: 在 `scripts/` 中编写 Python 脚本，将权重导出为简单的 **二进制文件 (.bin)**。
* C++ 端使用 `fread` 或 `mmap` 直接读取二进制数据到内存池中，这符合你设计的“权重手动导出”非目标。

### 4. 容错与调试 (Engineering Best Practices)

* **CUDA 检查宏**: 在第一周务必实现一个 `CHECK_CUDA(call)` 宏，包裹所有的 CUDA API。
* **性能 Profile**: 预留 `nvtx` 标记接口，这样你以后用 Nsight Systems 查看 Timeline 时，能清楚看到 `RMSNorm` -> `GEMM` -> `Attention` 的执行边界。

---

## 🧱 第一周：基础设施清单

在这一周，你可以按照以下顺序创建文件：

1. **`include/core/tensor.h`**: 定义 Tensor 类。
2. **`include/core/allocator.h`**: 定义你提到的栈分配器和块分配器。
3. **`include/utils/cuda_utils.h`**: 实现错误检查宏。
4. **`kernels/vector_add.cu`**: 编写一个简单的向量加法，验证 CMake 对 CUDA 的支持是否配置正确。

**你需要我为你提供一个标准的、支持 CUDA 构建的 `CMakeLists.txt` 模板作为起步吗？**