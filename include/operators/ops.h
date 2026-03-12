#pragma once
namespace tiny_llm {
    class Tensor;
    class ExecutionContext;

    namespace ops {
        // Minimal operator set for the current skeleton; extend incrementally.
        void rmsnorm(const Tensor& x, const Tensor& w, Tensor& y, ExecutionContext& ctx);
        void gemm(const Tensor& a, const Tensor& b, Tensor& c, ExecutionContext& ctx);

        // Paged-attention interface placeholder kept stable for integration.
        void attention_paged(const Tensor& q, Tensor& out, ExecutionContext& ctx);
    } // namespace ops
} // namespace tiny_llm