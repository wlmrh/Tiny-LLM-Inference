#include "tiny_llm/runtime/generation_config.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace
{

std::filesystem::path make_temp_dir(const std::string &name)
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("tinyllm_" + name + "_" + std::to_string(now));
    std::filesystem::create_directories(dir);
    return dir;
}

void write_text_file(const std::filesystem::path &path, const std::string &content)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("failed to open test file for write: " + path.string());
    }
    out << content;
}

} // namespace

TEST(GenerationConfigTest, ReturnsDefaultsWhenConfigFileIsMissing)
{
    const std::filesystem::path dir = make_temp_dir("generation_config_default");
    try
    {
        const tiny_llm::GenerationConfig config = tiny_llm::load_generation_config_from_dir(dir.string());
        EXPECT_FLOAT_EQ(config.repetition_penalty, 1.0f);
    }
    catch (...)
    {
        std::filesystem::remove_all(dir);
        throw;
    }
    std::filesystem::remove_all(dir);
}

TEST(GenerationConfigTest, ReadsRepetitionPenaltyFromModelDirectory)
{
    const std::filesystem::path dir = make_temp_dir("generation_config_penalty");
    try
    {
        write_text_file(dir / "generation_config.json", R"({"repetition_penalty": 1.15})");
        const tiny_llm::GenerationConfig config = tiny_llm::load_generation_config_from_dir(dir.string());
        EXPECT_FLOAT_EQ(config.repetition_penalty, 1.15f);
    }
    catch (...)
    {
        std::filesystem::remove_all(dir);
        throw;
    }
    std::filesystem::remove_all(dir);
}
