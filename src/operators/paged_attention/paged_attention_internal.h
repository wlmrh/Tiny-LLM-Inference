#pragma once

#include "tiny_llm/operators/paged_attention.h"

#include <cstdint>
#include <functional>
#include <initializer_list>

#if TINYLLM_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace tiny_llm
{
class KVCache;

namespace ops
{

int32_t kv_hidden_size(int32_t num_key_value_heads, int32_t head_dim);
int32_t attention_hidden_size(int32_t num_attention_heads, int32_t head_dim);

bool any_cuda(std::initializer_list<std::reference_wrapper<const Tensor>> tensors);
void validate_same_device(std::initializer_list<std::reference_wrapper<const Tensor>> tensors, const char *name);
Tensor tensor_to_cpu_contiguous(const Tensor &tensor);
const int32_t *cpu_int_ptr(const Tensor &tensor, Tensor &storage);

void validate_llama_attention_params(const LlamaAttentionParams &params);
void validate_paged_metadata_for_attention(const LlamaAttentionParams &params);
bool has_paged_kv_cache(const LlamaAttentionParams &params);

void run_direct_attention_cpu(const LlamaAttentionParams &params);
void run_paged_attention_cpu(const LlamaAttentionParams &params);
void run_torch_reference_attention(const LlamaAttentionParams &params);
bool try_run_cuda_optimized_attention(const LlamaAttentionParams &params);

#if TINYLLM_ENABLE_CUDA
namespace cuda
{
void launch_attention_paged_f32(const float *q, float *out, int64_t numel, cudaStream_t stream);
void launch_write_paged_kv_cache_f32(const float *k, const float *v, const int32_t *positions,
                                     const int32_t *seq_indices, const int32_t *block_tables, float *kv_pool_base,
                                     int64_t rows, int64_t num_seqs, int64_t max_blocks_per_seq, int64_t num_blocks,
                                     int64_t block_size_bytes, int32_t block_size_tokens, int32_t layer_id,
                                     int32_t kv_size, cudaStream_t stream);
void launch_paged_attention_query_f32(const float *q, float *out, const int32_t *positions, const int32_t *seq_indices,
                                      const int32_t *context_lens, const int32_t *block_tables,
                                      const float *kv_pool_base, int64_t rows, int64_t num_seqs,
                                      int64_t max_blocks_per_seq, int64_t num_blocks, int64_t block_size_bytes,
                                      int32_t block_size_tokens, int32_t layer_id, int32_t num_attention_heads,
                                      int32_t num_key_value_heads, int32_t head_dim, cudaStream_t stream);
void launch_paged_attention_f32(const float *q, const float *k, const float *v, float *out, const int32_t *positions,
                                const int32_t *seq_indices, const int32_t *context_lens, const int32_t *block_tables,
                                float *kv_pool_base, int64_t rows, int64_t num_seqs, int64_t max_blocks_per_seq,
                                int64_t num_blocks, int64_t block_size_bytes, int32_t block_size_tokens,
                                int32_t layer_id, int32_t num_attention_heads, int32_t num_key_value_heads,
                                int32_t head_dim, cudaStream_t stream);
void launch_paged_attention_bf16_kv(const float *q, const float *k, const float *v, float *out,
                                    const int32_t *positions, const int32_t *seq_indices, const int32_t *context_lens,
                                    const int32_t *block_tables, void *kv_pool_base, int64_t rows, int64_t num_seqs,
                                    int64_t max_blocks_per_seq, int64_t num_blocks, int64_t block_size_bytes,
                                    int32_t block_size_tokens, int32_t layer_id, int32_t num_attention_heads,
                                    int32_t num_key_value_heads, int32_t head_dim, cudaStream_t stream);
} // namespace cuda
#endif

} // namespace ops
} // namespace tiny_llm
