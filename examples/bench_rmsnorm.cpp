#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "tiny_llm/core/context.h"
#include "tiny_llm/core/tensor.h"
#include "tiny_llm/operators/rmsnorm.h"
#include "utils/cuda_utils.h"
#include <nvtx3/nvToolsExt.h>
namespace {
// input string that represents the value of varient name
int parse_int_arg(const char* s, const char* name)
{
  const int v = std::atoi(s);
  if (v <= 0)
  {
    throw std::runtime_error(std::string(name) + " must be positive.");
  }
  return v;
}

float parse_float_arg(const char* s, const char* name)
{
  const float v = std::strtof(s, nullptr);
  if (v <= 0.0f)
  {
    throw std::runtime_error(std::string(name) + " must be > 0.");
  }
  return v;
}

} // namespace

int main(int argc, char** argv)
{
  int B = 32;
  int D = 4096;
  int iters = 500;
  constexpr int kWarmupIters = 10;
  float eps = 1e-5f;

  // 读入输入参数
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg(argv[i]);
    if (arg == "--B" && i + 1 < argc)
    {
      B = parse_int_arg(argv[++i], "B");
    }
    else if (arg == "--D" && i + 1 < argc)
    {
      D = parse_int_arg(argv[++i], "D");
    }
    else if (arg == "--iters" && i + 1 < argc)
    {
      iters = parse_int_arg(argv[++i], "iters");
    }
    else if (arg == "--eps" && i + 1 < argc)
    {
      eps = parse_float_arg(argv[++i], "eps");
    }
    else
    {
      std::cerr << "Usage: " << argv[0]
                << " [--B <int>] [--D <int>] [--iters <200-1000>] [--eps <float>]\n";
      return 1;
    }
  }

  iters = std::clamp(iters, 200, 1000);

  const std::size_t x_numel = static_cast<std::size_t>(B) * static_cast<std::size_t>(D);
  const std::size_t w_numel = static_cast<std::size_t>(D);

  try
  {
    CHECK_CUDA(cudaSetDevice(0));

    std::vector<float> h_x(x_numel); // input matrix
    std::vector<float> h_w(w_numel); // weight vector

    // fill with random numbers
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < h_x.size(); ++i)
    {
      h_x[i] = dist(rng);
    }
    for (std::size_t i = 0; i < h_w.size(); ++i)
    {
      h_w[i] = dist(rng);
    }

    float* d_x = nullptr;
    float* d_w = nullptr;
    float* d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, x_numel * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, w_numel * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, x_numel * sizeof(float)));

    // dst, src, count, cudaMemcpyKind
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), x_numel * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), w_numel * sizeof(float), cudaMemcpyHostToDevice));

    // device Memory, value, count
    CHECK_CUDA(cudaMemset(d_y, 0, x_numel * sizeof(float)));

    tiny_llm::Tensor x(d_x, {B, D}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor w(d_w, {D}, tiny_llm::DType::kFloat32);
    tiny_llm::Tensor y(d_y, {B, D}, tiny_llm::DType::kFloat32);
    tiny_llm::ExecutionContext ctx(0, nullptr, nullptr);

    for (int i = 0; i < kWarmupIters; ++i)
    {
      tiny_llm::ops::rmsnorm(x, w, y, ctx, eps);
    }
    CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    // 创建一个开始、结束时间
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    nvtxRangePushA("rmsnorm_bench_timed_loop");
    // 插入记时开始事件
    CHECK_CUDA(cudaEventRecord(start, ctx.stream()));
    for (int i = 0; i < iters; ++i)
    {
      tiny_llm::ops::rmsnorm(x, w, y, ctx, eps);
    }
    // 插入记时结束事件
    CHECK_CUDA(cudaEventRecord(stop, ctx.stream()));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaGetLastError());

    const double avg_us = static_cast<double>(elapsed_ms) * 1000.0 / static_cast<double>(iters);
    // Approximate per-iteration traffic: read x twice (sumsq + writeback), read w once, write y once.
    const double bytes_per_iter = static_cast<double>(x_numel) * static_cast<double>(sizeof(float)) * 4.0;
    const double gbps = bytes_per_iter / (avg_us * 1e-6) / 1e9;

    std::cout << "bench_rmsnorm\n";
    std::cout << "  B=" << B << ", D=" << D << ", warmup=" << kWarmupIters
              << ", iters=" << iters << ", eps=" << eps << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  avg: " << avg_us << " us\n";
    std::cout << std::setprecision(2);
    std::cout << "  throughput: " << gbps << " GB/s\n";

    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_x));
  }
  catch (const std::exception& e)
  {
    std::cerr << "bench_rmsnorm failed: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
