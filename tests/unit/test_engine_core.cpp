#include "tiny_llm/core/allocator.h"
#include "tiny_llm/models/model.h"
#include "tiny_llm/runtime/engine.h"
#include "tiny_llm/runtime/engine_core.h"
#include "tiny_llm/runtime/engine_args.h"
#include "tiny_llm/runtime/processors.h"
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
    std::vector<int32_t> encode(const std::string&) override { return {4}; }
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
          core(make_engine_args(&model, &ctx, &kv, &tokenizer, max_prefill_tokens))
    {
    }

    static tiny_llm::EngineArgs make_engine_args(IncrementTokenModel* model,
                                                 tiny_llm::ExecutionContext* ctx,
                                                 tiny_llm::KVCache* kv,
                                                 FakeTokenizer* tokenizer,
                                                 int32_t max_prefill_tokens)
    {
        tiny_llm::EngineArgs args;
        args.model = model;
        args.ctx = ctx;
        args.kv = kv;
        args.tokenizer = tokenizer;
        args.max_generated_tokens = 8;
        args.scheduler_config = make_scheduler_config(max_prefill_tokens);
        return args;
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

    static tiny_llm::EngineArgs make_engine_args(tiny_llm::Model* model,
                                                tiny_llm::ExecutionContext* ctx,
                                                tiny_llm::KVCache* kv,
                                                tiny_llm::Tokenizer* tokenizer,
                                                int32_t max_prefill_tokens)
    {
        tiny_llm::EngineArgs args;
        args.model = model;
        args.ctx = ctx;
        args.kv = kv;
        args.tokenizer = tokenizer;
        args.max_generated_tokens = 8;
        args.scheduler_config = make_scheduler_config(max_prefill_tokens);
        return args;
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

TEST(InputPreprocessorTest, DefaultUserMaxTokensUsesEngineDefaultAndCopiesSharedFields)
{
    FakeTokenizer tokenizer;
    tiny_llm::EngineArgs args;
    args.tokenizer = &tokenizer;
    args.max_generated_tokens = 7;

    tiny_llm::InputPreprocessor preprocessor(args);
    tiny_llm::UserSamplingParams user_params;
    user_params.temperature = 0.8f;
    user_params.top_p = 0.7f;
    user_params.top_k = 3;
    user_params.repetition_penalty = 1.2f;
    user_params.seed = 42;
    user_params.ignore_eos = true;
    user_params.stop_token_ids = {6, 7};

    const tiny_llm::EngineCoreRequest request =
        preprocessor.process_inputs("prompt", user_params, "sampling-default");
    const tiny_llm::SamplingParams& params = request.sampling_params;

    EXPECT_FLOAT_EQ(params.temperature, user_params.temperature);
    EXPECT_FLOAT_EQ(params.top_p, user_params.top_p);
    EXPECT_EQ(params.top_k, user_params.top_k);
    EXPECT_FLOAT_EQ(params.repetition_penalty, user_params.repetition_penalty);
    EXPECT_EQ(params.seed, user_params.seed);
    EXPECT_TRUE(params.ignore_eos);
    EXPECT_EQ(params.stop_token_ids, user_params.stop_token_ids);
    EXPECT_EQ(params.max_tokens, 7);
}

TEST(InputPreprocessorTest, ExplicitUserMaxTokensOverridesEngineDefault)
{
    FakeTokenizer tokenizer;
    tiny_llm::EngineArgs args;
    args.tokenizer = &tokenizer;
    args.max_generated_tokens = 7;

    tiny_llm::InputPreprocessor preprocessor(args);
    tiny_llm::UserSamplingParams user_params;
    user_params.max_tokens = 5;

    const tiny_llm::EngineCoreRequest request =
        preprocessor.process_inputs("prompt", user_params, "sampling-override");

    EXPECT_EQ(request.sampling_params.max_tokens, 5);
}

TEST(InputPreprocessorTest, RejectsNegativeUserMaxTokens)
{
    FakeTokenizer tokenizer;
    tiny_llm::EngineArgs args;
    args.tokenizer = &tokenizer;
    args.max_generated_tokens = 7;

    tiny_llm::InputPreprocessor preprocessor(args);
    tiny_llm::UserSamplingParams user_params;
    user_params.max_tokens = -1;

    EXPECT_THROW(
        preprocessor.process_inputs("prompt", user_params, "sampling-negative"),
        std::runtime_error);
}

TEST(EngineCoreTest, EmitsPrefillSampleAsFirstGeneratedTokenThenDecodes)
{
    EngineCoreFixture fixture;
    fixture.add_request(1, {10, 11}, 2);

    auto [prefill_outputs, did_work] = fixture.core.step();
    EXPECT_TRUE(did_work);
    ASSERT_EQ(prefill_outputs.size(), 1u);
    EXPECT_EQ(prefill_outputs.front().internal_id, 1u);
    EXPECT_EQ(prefill_outputs.front().new_token_id, 12);

    auto [decode_outputs, did_decode] = fixture.core.step();
    EXPECT_TRUE(did_decode);
    ASSERT_EQ(decode_outputs.size(), 1u);
    EXPECT_EQ(decode_outputs.front().internal_id, 1u);
    EXPECT_EQ(decode_outputs.front().new_token_id, 13);

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
    EXPECT_EQ(second_outputs.front().internal_id, 1u);
    EXPECT_EQ(second_outputs.front().new_token_id, 23);

    auto [empty_outputs, more_work] = fixture.core.step();
    EXPECT_FALSE(more_work);
    EXPECT_TRUE(empty_outputs.empty());
}

TEST(EngineCoreTest, RejectsPromptTokenOutsideModelVocabulary)
{
    EngineCoreFixture fixture;
    EXPECT_THROW(fixture.add_request(1, {63, 64}, 1), std::runtime_error);
}

TEST(EngineCoreTest, LLMEngineReleasesFinishedExternalRequestIds)
{
    EngineCoreFixture fixture;
    tiny_llm::EngineArgs args;
    args.model = &fixture.model;
    args.ctx = &fixture.ctx;
    args.kv = &fixture.kv;
    args.tokenizer = &fixture.tokenizer;
    args.max_generated_tokens = 1;
    args.scheduler_config.max_prefill_tokens_per_step = 8;

    tiny_llm::LLMEngine engine(args);
    tiny_llm::UserSamplingParams sampling_params;
    sampling_params.max_tokens = 1;

    EXPECT_NO_THROW(engine.add_request("first", sampling_params, "reuse-id"));
    while (engine.has_unfinished_requests())
    {
        (void)engine.step();
    }

    EXPECT_NO_THROW(engine.add_request("second", sampling_params, "reuse-id"));
}
