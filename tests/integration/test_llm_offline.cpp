#include "tiny_llm/runtime/llm.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main()
{
    const char* model_dir = std::getenv("TINYLLM_HF_TINY_LLAMA_DIR");
    if (model_dir == nullptr || std::string(model_dir).empty())
    {
        std::cerr << "TINYLLM_HF_TINY_LLAMA_DIR is not set; skipping offline LLM integration test.\n";
        return 77;
    }

    try
    {
        tiny_llm::LLMOptions options(model_dir);

        tiny_llm::UserSamplingParams sampling_params;
        sampling_params.temperature = 0.0f;
        sampling_params.max_tokens = 4;

        tiny_llm::LLM llm(options);
        const std::vector<tiny_llm::CompletionOutput> outputs = llm.generate(std::vector<std::string>{"hello", "tiny llm inference"},
                                                                             sampling_params);
        if (outputs.size() != 2)
        {
            std::cerr << "expected two outputs, got " << outputs.size() << "\n";
            return 1;
        }
        for (const tiny_llm::CompletionOutput& output : outputs)
        {
            if (!output.finished)
            {
                std::cerr << "request did not finish for prompt: " << output.prompt << "\n";
                return 1;
            }
            if (output.token_ids.empty())
            {
                std::cerr << "empty generated token ids for prompt: " << output.prompt << "\n";
                return 1;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }

    return 0;
}
