import argparse
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_benchmark_suite import resolve_backends, resolved_scenario


def make_args(**overrides):
    values = {
        "backend": None,
        "tinyllm_binary": "/bin/true",
        "vllm_python": "/definitely/missing/python",
        "allow_backend_skip": False,
        "capacity_rps": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class BackendOrchestrationTest(unittest.TestCase):
    def test_missing_requested_backend_fails_by_default(self):
        with self.assertRaisesRegex(RuntimeError, "requested backend unavailable"):
            resolve_backends(make_args(), {"backends": ["vllm"]}, {})

    def test_missing_requested_backend_skips_only_when_allowed(self):
        selected, skipped = resolve_backends(
            make_args(allow_backend_skip=True), {"backends": ["vllm"]}, {}
        )
        self.assertEqual(selected, [])
        self.assertEqual(skipped, ["vllm"])

    def test_open_loop_rejects_batch_backends(self):
        with self.assertRaisesRegex(RuntimeError, "open-loop"):
            resolve_backends(
                make_args(),
                {"backends": ["tinyllm", "transformers"], "defaults": {"traffic_mode": "open-loop"}},
                {},
            )

    def test_fractional_rate_requires_capacity(self):
        with self.assertRaisesRegex(RuntimeError, "capacity-rps"):
            resolved_scenario(make_args(), {}, {"request_rate_fraction": 0.5})
        scenario = resolved_scenario(
            make_args(capacity_rps=12.0), {}, {"request_rate_fraction": 0.5}
        )
        self.assertEqual(scenario["request_rate_rps"], 6.0)


if __name__ == "__main__":
    unittest.main()
