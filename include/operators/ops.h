#pragma once
namespace tiny_llm {
    class Tensor;
    class ExecutionContext;
    namespace ops {
        // 初版先放最少的 op；后续再扩展
        void rmsnorm(const Tensor& x, const Tensor& w, Tensor& y, ExecutionContext& ctx);
        void gemm(const Tensor& a, const Tensor& b, Tensor& c, ExecutionContext& ctx);

        // paged attention 初版可以先 stub，只把接口定住
        void attention_paged(const Tensor& q, /*...*/, Tensor& out, ExecutionContext& ctx);
    } // namespace ops
} // namespace tiny_llm