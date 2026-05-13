#include "tiny_llm/core/allocator.h"
#include "tiny_llm/models/model.h"
#include "tiny_llm/runtime/engine_core.h"
#include "tiny_llm/runtime/kv_cache.h"
#include "tiny_llm/runtime/tokenizer.h"

#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
class IncrementTokenModel final : public tiny_llm::Model {
public:
    int32_t num_layers() const override { return 1; }
    int32_t vocab_size() const override { return 64; }

    tiny_llm::Tensor forward(const tiny_llm::PreparedInputs& inputs,
                             tiny_llm::RuntimeContext& ctx) override
    {
        tiny_llm::Tensor input_cpu = inputs.input_ids.cpu().contiguous();
        tiny_llm::Tensor logits = torch::full(
            {inputs.input_ids.size(0), vocab_size()},
            -1000.0f,
            torch::TensorOptions().dtype(torch::kFloat32).device(c10::kCPU));
        const int32_t* input = input_cpu.data_ptr<int32_t>();
        float* out = logits.data_ptr<float>();
        for (int64_t row = 0; row < input_cpu.numel(); ++row)
        {
            out[row * vocab_size() + ((input[row] + 1) % vocab_size())] = 1000.0f;
        }
        return logits.to(ctx.device());
    }
};

class FakeTokenizer final : public tiny_llm::Tokenizer {
public:
    std::vector<int32_t> encode(const std::string&) override { return {}; }
    std::string decode(const std::vector<int32_t>& ids) const override
    {
        std::string result;
        for (int32_t id : ids)
        {
            result += std::to_string(id);
        }
        return result;
    }
    int32_t vocab_size() const override { return 64; }
    int32_t bos_id() const override { return 1; }
    int32_t eos_id() const override { return 2; }
    int32_t unk_id() const override { return 3; }
    bool is_fixed_vocab() const override { return true; }
    bool is_valid_token_id(int32_t id) const override { return id >= 0 && id < vocab_size(); }
};

struct EngineCoreFixture {
    static constexpr size_t kBlockBytes = 128;
    std::vector<unsigned char> pool;
    tiny_llm::BlockAllocator blocks;
    tiny_llm::KVCache kv;
    IncrementTokenModel model;
    FakeTokenizer tokenizer;
    tiny_llm::ExecutionContext ctx;
    tiny_llm::EngineCore core;

    explicit EngineCoreFixture(int32_t max_prefill_tokens = 8, size_t num_blocks = 32)
        : pool(num_blocks * kBlockBytes),
          blocks(num_blocks, kBlockBytes, pool.data(), tiny_llm::ParallelConfig::cpu()),
          kv(make_kv_config(), &blocks),
          ctx(nullptr, nullptr, &kv),
          core(&model, &ctx, &kv, &tokenizer, 8, make_scheduler_config(max_prefill_tokens))
    {
    }

    static tiny_llm::KVCache::Config make_kv_config()
    {
        tiny_llm::KVCache::Config cfg;
        cfg.num_layers = 1;
        cfg.block_size_tokens = 2;
        return cfg;
    }

    static tiny_llm::SchedulerConfig make_scheduler_config(int32_t max_prefill_tokens)
    {
        tiny_llm::SchedulerConfig cfg;
        cfg.max_prefill_tokens_per_step = max_prefill_tokens;
        return cfg;
    }

    void add_request(uint64_t id, std::vector<int32_t> prompt, int32_t max_tokens)
    {
        tiny_llm::EngineCoreRequest request;
        request.internal_id = id;
        request.prompt_token_ids = std::move(prompt);
        request.sampling_params.max_tokens = max_tokens;
        core.add_request(request);
    }
};
}

TEST(EngineCoreTest, EmitsPrefillSampleAsFirstGeneratedTokenThenDecodes)
{
    EngineCoreFixture fixture;
    fixture.add_request(1, {10, 11}, 2);

    auto [prefill_outputs, did_work] = fixture.core.step();
    EXPECT_TRUE(did_work);
    ASSERT_EQ(prefill_outputs.size(), 1u);
    EXPECT_EQ(prefill_outputs.at(1).new_token_id, 12);
    EXPECT_EQ(prefill_outputs.at(1).generated_tokens, 1);

    auto [decode_outputs, did_decode] = fixture.core.step();
    EXPECT_TRUE(did_decode);
    ASSERT_EQ(decode_outputs.size(), 1u);
    EXPECT_EQ(decode_outputs.at(1).new_token_id, 13);
    EXPECT_EQ(decode_outputs.at(1).generated_tokens, 2);

    auto [empty_outputs, more_work] = fixture.core.step();
    EXPECT_FALSE(more_work);
    EXPECT_TRUE(empty_outputs.empty());
}

TEST(EngineCoreTest, DoesNotEmitOutputUntilChunkedPrefillCompletes)
{
    EngineCoreFixture fixture(2);
    fixture.add_request(1, {20, 21, 22}, 1);

    auto [first_outputs, first_work] = fixture.core.step();
    EXPECT_TRUE(first_work);
    EXPECT_TRUE(first_outputs.empty());

    auto [second_outputs, second_work] = fixture.core.step();
    EXPECT_TRUE(second_work);
    ASSERT_EQ(second_outputs.size(), 1u);
    EXPECT_EQ(second_outputs.at(1).new_token_id, 23);
    EXPECT_EQ(second_outputs.at(1).generated_tokens, 1);

    auto [empty_outputs, more_work] = fixture.core.step();
    EXPECT_FALSE(more_work);
    EXPECT_TRUE(empty_outputs.empty());
}

TEST(EngineCoreTest, RejectsPromptTokenOutsideModelVocabulary)
{
    EngineCoreFixture fixture;
    EXPECT_THROW(fixture.add_request(1, {63, 64}, 1), std::runtime_error);
}
