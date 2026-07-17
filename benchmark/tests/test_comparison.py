import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_benchmark_comparison import build_comparison


def backend_result(name, token_ids):
    return {
        "backend": name,
        "prompt_count": 1,
        "max_new_tokens": 2,
        "avg_generated_tokens": 2.0,
        "samples": [{"request_id": "r0", "token_ids": token_ids}],
        "avg_total_latency_ms": 1.0,
        "avg_first_token_latency_ms": 1.0,
        "end_to_end_tokens_per_s": 1.0,
        "decode_tokens_per_s": 1.0,
        "avg_load_init_ms": 1.0,
    }


class ComparisonTest(unittest.TestCase):
    def test_token_agreement_records_mismatches(self):
        comparison = build_comparison(
            [backend_result("tinyllm", [1, 2]), backend_result("transformers", [1, 3])]
        )
        agreement = comparison["output_agreement"]["tinyllm_vs_transformers"]
        self.assertFalse(agreement["match"])
        self.assertEqual(agreement["mismatch_count"], 1)
        self.assertEqual(agreement["mismatched_request_ids"], ["r0"])


if __name__ == "__main__":
    unittest.main()
