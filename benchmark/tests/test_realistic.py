import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suite.metrics import enrich_and_group_requests, relative_goodput
from suite.realistic import (
    CandidateMatcher,
    load_burstgpt_rows,
    load_oasst_candidates,
    scale_trace_arrivals,
    select_trace_windows,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize
        text = " ".join(f"{item['role']} {item['content']}" for item in messages)
        return text + (" assistant" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return list(range(len(text.split())))


class RealisticWorkloadTest(unittest.TestCase):
    def test_oasst_tree_reconstruction_ends_with_user(self):
        tree = {
            "message_id": "root",
            "role": "prompter",
            "lang": "en",
            "text": "one two",
            "replies": [
                {
                    "message_id": "answer",
                    "role": "assistant",
                    "lang": "en",
                    "text": "three four",
                    "replies": [
                        {
                            "message_id": "followup",
                            "role": "prompter",
                            "lang": "en",
                            "text": "five six",
                            "replies": [],
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trees.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(tree) + "\n")
            candidates = load_oasst_candidates(path, FakeTokenizer())
        self.assertEqual([item["message_id"] for item in candidates], ["root", "followup"])
        self.assertTrue(all(item["prompt"].endswith("assistant") for item in candidates))

    def test_burstgpt_filter_and_time_order(self):
        rows = [
            ["2", "s2", "ChatGPT", "10", "20", "Conversation log"],
            ["1", "s1", "ChatGPT", "10", "600", "API log"],
            ["0", "s0", "ChatGPT", "12", "8", "API log"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Timestamp", "Session ID", "Model", "Request tokens", "Response tokens", "Log Type"])
                writer.writerows(rows)
            loaded = load_burstgpt_rows(path)
        self.assertEqual([item["source_timestamp_s"] for item in loaded], [0.0, 2.0])

    def test_matching_is_unique_and_deterministic(self):
        candidates = [
            {"prompt_len": length, "message_id": f"m{length:03d}", "prompt": str(length), "input_ids": [],
             "tree_id": "t", "language": "en"}
            for length in range(1, 80)
        ]
        matcher = CandidateMatcher(candidates)
        first = matcher.match(20, 100)
        second = matcher.match(20, 100)
        self.assertEqual(first["prompt_len"], 20)
        self.assertNotEqual(first["message_id"], second["message_id"])

    def test_window_selection_and_arrival_scaling(self):
        rows = [
            {
                "source_trace_index": index,
                "source_timestamp_s": float(index),
                "source_request_tokens": 20 + index % 5,
                "source_response_tokens": 8,
                "source_log_type": "API log" if index % 2 else "Conversation log",
                "source_model": "ChatGPT",
                "source_session_sha256": str(index),
            }
            for index in range(60)
        ]
        candidates = [
            {"prompt_len": 20 + index % 5, "message_id": f"m{index:03d}", "prompt": f"p{index}",
             "input_ids": [1] * (20 + index % 5), "tree_id": f"t{index}", "language": "en"}
            for index in range(60)
        ]
        windows = select_trace_windows(rows, candidates, window_size=4)
        trace_ids = [item["source_trace_index"] for window in windows for item in window]
        self.assertEqual(len(trace_ids), len(set(trace_ids)))
        scaled = scale_trace_arrivals(windows[0], 2.0)
        self.assertAlmostEqual(scaled[-1]["arrival_ms"], 1500.0)
        self.assertTrue(all(a["arrival_ms"] <= b["arrival_ms"] for a, b in zip(scaled, scaled[1:])))

    def test_grouped_metrics_and_relative_goodput(self):
        requests = [
            {"request_id": "r0", "generated_tokens": 8, "finish_ms": 100.0, "ttft_ms": 10.0,
             "tpot_ms": 5.0, "queue_ms": 1.0, "engine_ttft_ms": 9.0, "e2e_ms": 100.0},
            {"request_id": "r1", "generated_tokens": 8, "finish_ms": 200.0, "ttft_ms": 30.0,
             "tpot_ms": 15.0, "queue_ms": 2.0, "engine_ttft_ms": 28.0, "e2e_ms": 200.0},
        ]
        workload = [
            {"request_id": "r0", "prompt_len": 64, "max_new_tokens": 8, "source_log_type": "API log"},
            {"request_id": "r1", "prompt_len": 600, "max_new_tokens": 8, "source_log_type": "Conversation log"},
        ]
        grouped = enrich_and_group_requests(requests, workload)
        self.assertEqual(grouped["overall"]["request_count"], 2)
        self.assertIn("isl:1-128", grouped["groups"])
        goodput = relative_goodput(grouped["requests"], 20.0, 10.0)
        self.assertEqual(goodput["good_requests"], 1.0)


if __name__ == "__main__":
    unittest.main()
