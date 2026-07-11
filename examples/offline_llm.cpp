#include "tiny_llm/runtime/llm.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

tiny_llm::ParallelConfig parse_device(const std::string &value)
{
    if (value == "cpu")
    {
        return tiny_llm::ParallelConfig::cpu();
    }
    if (value == "cuda")
    {
        return tiny_llm::ParallelConfig::cuda(0);
    }
    const std::string prefix = "cuda:";
    if (value.rfind(prefix, 0) == 0)
    {
        return tiny_llm::ParallelConfig::cuda(std::stoi(value.substr(prefix.size())));
    }
    throw std::runtime_error("offline_llm: expected device cpu, cuda, or cuda:<id>.");
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const char *env_model = std::getenv("TINYLLM_HF_TINY_LLAMA_DIR");
        const std::string model = argc > 1 ? argv[1] : (env_model != nullptr ? env_model : "/models/smollm2-135M");
        const std::string device = argc > 2 ? argv[2] : "cpu";

        tiny_llm::LLMOptions options(model, parse_device(device));

        tiny_llm::LLMSamplingParams sampling_params;
        sampling_params.temperature = 0.8f;
        sampling_params.top_p = 0.95f;

        tiny_llm::LLM llm(options);
        const std::vector<std::string> prompts = {
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        };

        const std::vector<tiny_llm::CompletionOutput> outputs = llm.generate(prompts, sampling_params);

        std::cout << "\nGenerated Outputs:\n";
        std::cout << "------------------------------------------------------------\n";
        for (const tiny_llm::CompletionOutput &output : outputs)
        {
            std::cout << "Prompt:    " << output.prompt << "\n";
            std::cout << "Output:    " << output.text << "\n";
            std::cout << "------------------------------------------------------------\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }

    return 0;
}
