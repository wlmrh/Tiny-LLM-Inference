import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suite.metrics import enrich_and_group_requests, relative_goodput
from suite.realistic import (
    CandidateMatcher,
    load_burstgpt_rows,
    load_oasst_candidates,
    public_selection,
    scale_trace_arrivals,
    select_trace_windows,
    sha256_text,
)
from run_realistic_benchmark import build_public_manifest, build_reproduction_commands, parse_args
from transformers_generate_benchmark import trim_generated_tokens_at_eos


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize
        text = " ".join(f"{item['role']} {item['content']}" for item in messages)
        return text + (" assistant" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return list(range(len(text.split())))


class RealisticWorkloadTest(unittest.TestCase):
    def test_realistic_cli_uses_current_python_unless_overridden(self):
        required = ["run_realistic_benchmark.py", "--prepared-dir", "prepared", "--output-dir", "output"]
        with patch.object(sys, "argv", required):
            default_args = parse_args()
        with patch.object(sys, "argv", [*required, "--vllm-python", "/opt/vllm/bin/python"]):
            explicit_args = parse_args()

        self.assertEqual(default_args.vllm_python, sys.executable)
        self.assertEqual(explicit_args.vllm_python, "/opt/vllm/bin/python")

    def test_reproduction_commands_use_portable_paths_and_resolved_options(self):
        args = type(
            "Args",
            (),
            {
                "config": "benchmark/configs/custom.json",
                "config_data": {
                    "sources": {
                        "burstgpt": {"filename": "trace.csv"},
                        "oasst1": {"filename": "trees.jsonl.gz"},
                    }
                },
                "tinyllm_binary": "build/cuda-release/benchmark/llama_engine_benchmark",
                "vllm_python": "/private/venvs/vllm/bin/python",
                "artifact_checksum_file": "/private/artifacts/checksum.txt",
                "phase": "all",
                "resume": True,
            },
        )()

        commands = build_reproduction_commands(args)

        self.assertIn("benchmark/configs/custom.json", commands["prepare"])
        self.assertIn("<dataset-root>/raw/burstgpt/trace.csv", commands["prepare"])
        self.assertIn("<model-dir>", commands["run"])
        self.assertIn("<python-with-vllm>", commands["run"])
        self.assertIn("build/cuda-release/benchmark/llama_engine_benchmark", commands["run"])
        self.assertIn("<artifact-checksum-file>", commands["run"])
        self.assertIn("--resume", commands["run"])
        self.assertNotIn("/private/", commands["prepare"] + commands["run"])
        self.assertNotIn("/root/autodl-tmp", commands["prepare"] + commands["run"])

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
        repeated = select_trace_windows(rows, candidates, window_size=4)
        trace_ids = [item["source_trace_index"] for window in windows for item in window]
        self.assertEqual(len(trace_ids), len(set(trace_ids)))
        first_selection = json.dumps(
            [public_selection(window) for window in windows], sort_keys=True, separators=(",", ":")
        )
        second_selection = json.dumps(
            [public_selection(window) for window in repeated], sort_keys=True, separators=(",", ":")
        )
        self.assertEqual(sha256_text(first_selection), sha256_text(second_selection))
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
        self.assertEqual(grouped["overall"]["completed_request_count"], 2)
        self.assertEqual(grouped["overall"]["max_concurrency"], 2)
        self.assertEqual(grouped["overall"]["total_tokens_per_s"], 3400.0)
        self.assertIn("isl:1-128", grouped["groups"])
        goodput = relative_goodput(grouped["requests"], 20.0, 10.0)
        self.assertEqual(goodput["good_requests"], 1.0)

    def test_transformers_batch_padding_is_trimmed_after_eos(self):
        self.assertEqual(trim_generated_tokens_at_eos([10, 11, 99, 99], 99, False), [10, 11, 99])
        self.assertEqual(trim_generated_tokens_at_eos([10, 11, 99, 99], 99, True), [10, 11, 99, 99])
        self.assertEqual(trim_generated_tokens_at_eos([10, 11], 99, False), [10, 11])

    def test_public_manifest_records_pinned_sources_without_private_dataset_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            burstgpt = root / "trace.csv"
            oasst1 = root / "trees.jsonl.gz"
            burstgpt.write_text("trace", encoding="utf-8")
            oasst1.write_text("oasst", encoding="utf-8")
            model_dir = root / "model"
            model_dir.mkdir()
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                (model_dir / name).write_text(name, encoding="utf-8")
            sources = {
                "burstgpt": {
                    "repo_id": "HPMLL/BurstGPT",
                    "revision": "burst-revision",
                    "sha256": sha256_text("trace"),
                },
                "oasst1": {
                    "repo_id": "OpenAssistant/oasst1",
                    "revision": "oasst-revision",
                    "sha256": sha256_text("oasst"),
                },
                "model": {"repo_id": "Qwen/model", "revision": "model-revision"},
            }
            prepared = {
                "burstgpt": {"path": str(burstgpt), "sha256": sha256_text("trace")},
                "oasst1": {"path": str(oasst1), "sha256": sha256_text("oasst")},
                "model_dir": str(model_dir),
            }
            manifest = build_public_manifest(prepared, sources, model_dir)

        self.assertEqual(manifest["burstgpt"]["revision"], "burst-revision")
        self.assertEqual(manifest["burstgpt"]["filename"], "trace.csv")
        self.assertNotIn("path", manifest["burstgpt"])
        self.assertEqual(manifest["oasst1"]["revision"], "oasst-revision")
        self.assertEqual(manifest["model"]["revision"], "model-revision")
        self.assertEqual(len(manifest["model"]["files"]), 3)
        self.assertNotIn("model_dir", manifest)


if __name__ == "__main__":
    unittest.main()
