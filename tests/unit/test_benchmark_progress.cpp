#include "../../benchmark/progress_guard.h"

#include <gtest/gtest.h>

TEST(BenchmarkProgressGuardTest, RejectsZeroScheduledTokensOnlyWhileRequestsRemain)
{
    EXPECT_NO_THROW(tiny_llm::benchmark::require_step_progress(false, 0));
    EXPECT_NO_THROW(tiny_llm::benchmark::require_step_progress(true, 1));
    EXPECT_THROW(tiny_llm::benchmark::require_step_progress(true, 0), std::runtime_error);
}
