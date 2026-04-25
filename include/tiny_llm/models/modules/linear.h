#pragma once

#include <cstdint>

#include "tiny_llm/core/tensor.h"

namespace tiny_llm {

class ExecutionContext;

namespace modules {

enum class WeightLayout {
    kInOut = 0,
    kOutIn = 1,
};

struct StackedWeightDesc {
    float* data = nullptr;
    int32_t out_features = 0;
    int32_t in_features = 0;
    int32_t output_offset = 0;
    WeightLayout layout = WeightLayout::kInOut;
};

class Linear {
public:
    Linear(int32_t in_features, int32_t out_features_total);

    void bind_weight(float* weight,
                     int32_t out_features,
                     int32_t in_features,
                     WeightLayout layout = WeightLayout::kInOut);
    void bind_stacked_weights(const StackedWeightDesc* descs, int32_t count);
    void forward(const Tensor& input, Tensor& output, ExecutionContext& ctx) const;

    int32_t in_features() const { return in_features_; }
    int32_t out_features_total() const { return out_features_total_; }

private:
    void validate_forward_inputs(const Tensor& input, const Tensor& output) const;
    void validate_descs(const StackedWeightDesc* descs, int32_t count) const;

    StackedWeightDesc single_weight_{};
    const StackedWeightDesc* stacked_weights_ = nullptr;
    int32_t stacked_weight_count_ = 0;
    bool use_stacked_weights_ = false;
    int32_t in_features_ = 0;
    int32_t out_features_total_ = 0;
};

} // namespace modules
} // namespace tiny_llm
