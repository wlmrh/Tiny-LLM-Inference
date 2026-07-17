import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suite.metrics import percentile, read_event_metrics


class MetricsTest(unittest.TestCase):
    def test_percentile_ordering(self):
        values = [9.0, 1.0, 5.0, 3.0, 7.0]
        self.assertLessEqual(percentile(values, 0.50), percentile(values, 0.95))
        self.assertLessEqual(percentile(values, 0.95), percentile(values, 0.99))

    def test_complete_event_metrics(self):
        events = [
            {"repeat": 0, "request_id": "r0", "prompt_index": 0, "event": "submit", "time_ms": 10.0},
            {"repeat": 0, "request_id": "r0", "prompt_index": 0, "event": "admit", "time_ms": 12.0},
            {"repeat": 0, "request_id": "r0", "prompt_index": 0, "event": "token", "time_ms": 20.0},
            {"repeat": 0, "request_id": "r0", "prompt_index": 0, "event": "token", "time_ms": 24.0},
            {
                "repeat": 0,
                "request_id": "r0",
                "prompt_index": 0,
                "event": "finish",
                "time_ms": 24.0,
                "generated_tokens": 2,
                "finish_reason": "length",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            metrics = read_event_metrics(path)

        self.assertTrue(metrics["available"])
        self.assertTrue(metrics["complete"])
        self.assertEqual(len(metrics["requests"]), 1)
        request = metrics["requests"][0]
        self.assertEqual(request["queue_ms"], 2.0)
        self.assertEqual(request["ttft_ms"], 10.0)
        self.assertEqual(request["engine_ttft_ms"], 8.0)
        self.assertEqual(request["tpot_ms"], 4.0)
        self.assertEqual(request["e2e_ms"], 14.0)
        for summary in metrics["summary"].values():
            self.assertLessEqual(summary["p50"], summary["p95"])
            self.assertLessEqual(summary["p95"], summary["p99"])

    def test_missing_admit_is_reported(self):
        events = [
            {"repeat": 0, "request_id": "r0", "event": "submit", "time_ms": 0.0},
            {"repeat": 0, "request_id": "r0", "event": "finish", "time_ms": 1.0, "generated_tokens": 0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            metrics = read_event_metrics(path)
        self.assertFalse(metrics["complete"])
        self.assertTrue(any("admit" in error for error in metrics["completeness_errors"]))


if __name__ == "__main__":
    unittest.main()
