#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "tiny_llm/core/tensor.h"
#include "tiny_llm/models/llama_config.h"
#include "tiny_llm/runtime/parallel_config.h"

namespace tiny_llm {

class HFSafeTensorLoader;

/**
 * @brief Name-to-weight registry used by model layers.
 *
 * Raw pointers are exposed for zero-copy binding while Tensor handles are
 * retained internally to keep backing storage alive for the whole model.
 */
class WeightMap {
public:
    static WeightMap from_safetensors(const HFSafeTensorLoader& loader);
    static WeightMap from_safetensors(const HFSafeTensorLoader& loader,
                                      const ParallelConfig& parallel_config);

    void add_tensor(const std::string& name, const Tensor& tensor);
    void add_tensor(const std::string& name,
                    void* data,
                    const std::vector<int64_t>& shape,
                    DType dtype);

    bool contains(const std::string& name) const;
    void* get_tensor(const std::string& name) const;
    const Tensor& get_tensor_view(const std::string& name) const;
    std::vector<std::string> keys() const;

    template <typename T>
    T* get_tensor_as(const std::string& name) const
    {
        return static_cast<T*>(get_tensor(name));
    }

private:
    std::unordered_map<std::string, void*> tensor_ptrs_;
    std::unordered_map<std::string, Tensor> tensor_views_;
};

struct LlamaLayerWeights {
    Tensor input_layernorm;
    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;
    Tensor post_attention_layernorm;
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
};

struct MiniLLaMAWeights {
    Tensor embed_tokens;
    std::vector<LlamaLayerWeights> layers;
    Tensor norm;
    Tensor lm_head;
};

MiniLLaMAWeights load_llama_weights(const HFSafeTensorLoader& loader,
                                    const LlamaConfig& config);
MiniLLaMAWeights load_llama_weights(const HFSafeTensorLoader& loader,
                                    const LlamaConfig& config,
                                    const ParallelConfig& parallel_config);

MiniLLaMAWeights load_llama_weights(const WeightMap& weight_map,
                                    const LlamaConfig& config);

} // namespace tiny_llm
