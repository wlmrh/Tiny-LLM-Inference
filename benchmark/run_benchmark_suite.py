#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from suite.metrics import read_event_metrics
from suite.reporting import write_reports
from suite.workloads import load_tokenizer, write_workload_jsonl


DEFAULT_CONFIG = Path("benchmark/configs/qwen25_quick.json")


def parse_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TinyLLM benchmark suite")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-dir", help="override config model_dir")
    parser.add_argument("--device", help="override config device")
    parser.add_argument("--tinyllm-binary", help="override config tinyllm_binary")
    parser.add_argument("--backend", help="comma-separated backend override: tinyllm,transformers,vllm")
    parser.add_argument("--output-dir", default="benchmark/results")
    parser.add_argument("--label", help="report label; defaults to config label plus timestamp")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile-detail", action="store_true")
    parser.add_argument("--allow-backend-skip", action="store_true", default=True)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def selected_backends(args: argparse.Namespace, config: Dict[str, Any], scenario: Dict[str, Any]) -> List[str]:
    if args.backend:
        backends = parse_csv(args.backend)
    else:
        backends = list(scenario.get("backends", config.get("backends", ["tinyllm", "transformers"])))
    if "vllm" in backends and importlib.util.find_spec("vllm") is None and args.allow_backend_skip:
        return [backend for backend in backends if backend != "vllm"]
    return backends


def skipped_backends(args: argparse.Namespace, config: Dict[str, Any], scenario: Dict[str, Any]) -> List[str]:
    requested = parse_csv(args.backend) if args.backend else list(
        scenario.get("backends", config.get("backends", ["tinyllm", "transformers"]))
    )
    selected = selected_backends(args, config, scenario)
    return [backend for backend in requested if backend not in selected]


def parse_json_line(stdout: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text.startswith("{"):
            continue
        data, _ = decoder.raw_decode(text)
        return data
    raise RuntimeError("benchmark command did not emit a final JSON object")


def run_command(command: List[str], dry_run: bool) -> Dict[str, Any]:
    if dry_run:
        print("dry-run:", " ".join(command))
        return {"dry_run": True, "command": command, "results": [], "ratios": {}, "ratio_status": {}}
    run = subprocess.run(command, check=False, capture_output=True, text=True)
    if run.returncode != 0:
        sys.stderr.write(run.stdout)
        sys.stderr.write(run.stderr)
        raise RuntimeError(f"command failed with exit code {run.returncode}: {' '.join(command)}")
    return parse_json_line(run.stdout)


def command_for(
    args: argparse.Namespace,
    config: Dict[str, Any],
    scenario: Dict[str, Any],
    model_dir: Path,
    workload_path: Path,
    events_path: Path,
    backends: List[str],
) -> List[str]:
    defaults = dict(config.get("defaults", {}))
    warmup = int(scenario.get("warmup", defaults.get("warmup", 0)))
    repeat = int(scenario.get("repeat", defaults.get("repeat", 1)))
    mode = str(scenario.get("benchmark_mode", defaults.get("benchmark_mode", "fixed_output_perf")))
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("run_benchmark_comparison.py")),
        "--tinyllm-binary",
        str(args.tinyllm_binary or config.get("tinyllm_binary", "build-cuda/benchmark/llama_engine_benchmark")),
        "--backend",
        ",".join(backends),
        "--device",
        str(args.device or config.get("device", "cuda:0")),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--max-new-tokens",
        str(int(scenario["output_tokens"])),
        "--benchmark-mode",
        mode,
        "--workload-jsonl",
        str(workload_path),
        "--events-jsonl",
        str(events_path),
        "--temperature",
        str(float(scenario.get("temperature", defaults.get("temperature", 0.0)))),
        "--top-p",
        str(float(scenario.get("top_p", defaults.get("top_p", 1.0)))),
        "--top-k",
        str(int(scenario.get("top_k", defaults.get("top_k", 0)))),
        "--repetition-penalty",
        str(float(scenario.get("repetition_penalty", defaults.get("repetition_penalty", 1.0)))),
        "--seed",
        str(int(scenario.get("seed", defaults.get("seed", 0)))),
        "--json",
    ]
    if bool(scenario.get("ignore_eos", defaults.get("ignore_eos", False))):
        command.append("--ignore-eos")
    if args.profile_detail or bool(defaults.get("profile_detail", False)):
        command.append("--profile-detail")
    command.append(str(model_dir))
    return command


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    model_dir = Path(args.model_dir or config.get("model_dir", "/models/Qwen2.5-1.5B-Instruct")).expanduser()
    output_dir = Path(args.output_dir)
    timestamp = dt.datetime.now(dt.timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    label = args.label or f"{config.get('label', 'tinyllm_suite')}_{timestamp}"
    run_dir = output_dir / label
    workload_dir = run_dir / "workloads"
    events_dir = run_dir / "events"
    tokenizer = None if args.dry_run else load_tokenizer(model_dir)

    report: Dict[str, Any] = {
        "label": label,
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "model_dir": str(model_dir),
        "device": str(args.device or config.get("device", "cuda:0")),
        "scenarios": [],
    }

    for scenario in config.get("workloads", []):
        workload_path = workload_dir / f"{scenario['name']}.jsonl"
        workload_meta = (
            {"path": str(workload_path), "dry_run": True}
            if args.dry_run
            else write_workload_jsonl(workload_path, tokenizer, scenario, dict(config.get("defaults", {})))
        )
        backends = selected_backends(args, config, scenario)
        events_path = events_dir / f"{scenario['name']}.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        scenario_report: Dict[str, Any] = {
            "workload": scenario,
            "workload_jsonl": str(workload_path),
            "workload_meta": workload_meta,
            "benchmark_mode": str(scenario.get("benchmark_mode", config.get("defaults", {}).get("benchmark_mode", "-"))),
            "command": [],
            "selected_backends": backends,
            "skipped_backends": skipped_backends(args, config, scenario),
        }
        if not backends:
            scenario_report["error"] = "all requested backends were skipped"
            report["scenarios"].append(scenario_report)
            continue
        command = command_for(args, config, scenario, model_dir, workload_path, events_path, backends)
        scenario_report["command"] = command
        try:
            comparison = run_command(command, args.dry_run)
            scenario_report["results"] = comparison.get("results", [])
            scenario_report["ratios"] = comparison.get("ratios", {})
            scenario_report["ratio_status"] = comparison.get("ratio_status", {})
            scenario_report["tinyllm_events"] = read_event_metrics(events_path)
        except Exception as exc:
            scenario_report["error"] = str(exc)
        report["scenarios"].append(scenario_report)

    paths = write_reports(run_dir, label, report)
    print(f"wrote JSON report: {paths['json']}")
    print(f"wrote Markdown report: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"run_benchmark_suite failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
