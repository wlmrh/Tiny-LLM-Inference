#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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
    parser.add_argument("--vllm-python", help="Python executable used for the vLLM backend")
    parser.add_argument("--backend", help="comma-separated backend override: tinyllm,transformers,vllm")
    parser.add_argument("--output-dir", default="benchmark/results")
    parser.add_argument("--label", help="report label; defaults to config label plus timestamp")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile-detail", action="store_true")
    parser.add_argument("--capacity-rps", type=float, help="measured TinyLLM capacity for fractional open-loop loads")
    parser.add_argument("--allow-backend-skip", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def requested_backends(args: argparse.Namespace, config: Dict[str, Any], scenario: Dict[str, Any]) -> List[str]:
    if args.backend:
        return parse_csv(args.backend)
    return list(scenario.get("backends", config.get("backends", ["tinyllm", "transformers"])))


def python_module_available(executable: str, module: str) -> bool:
    try:
        run = subprocess.run(
            [executable, "-c", f"import {module}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return run.returncode == 0


def resolve_backends(
    args: argparse.Namespace, config: Dict[str, Any], scenario: Dict[str, Any]
) -> tuple[List[str], List[str]]:
    requested = requested_backends(args, config, scenario)
    unknown = [backend for backend in requested if backend not in {"tinyllm", "transformers", "vllm"}]
    if unknown:
        raise RuntimeError(f"unknown requested backends: {', '.join(unknown)}")

    traffic_mode = str(scenario.get("traffic_mode", config.get("defaults", {}).get("traffic_mode", "offline")))
    if traffic_mode == "open-loop" and requested != ["tinyllm"]:
        raise RuntimeError("open-loop scenarios must request only the TinyLLM backend")

    tinyllm_binary = Path(args.tinyllm_binary or config.get("tinyllm_binary", "build-cuda/benchmark/llama_engine_benchmark"))
    vllm_python = str(args.vllm_python or config.get("vllm_python", sys.executable))
    unavailable: Dict[str, str] = {}
    if "tinyllm" in requested and not tinyllm_binary.is_file():
        unavailable["tinyllm"] = f"benchmark binary does not exist: {tinyllm_binary}"
    if "transformers" in requested and not python_module_available(sys.executable, "transformers"):
        unavailable["transformers"] = f"transformers is unavailable in {sys.executable}"
    if "vllm" in requested and not python_module_available(vllm_python, "vllm"):
        unavailable["vllm"] = f"vllm is unavailable in {vllm_python}"

    if unavailable and not args.allow_backend_skip:
        details = "; ".join(f"{backend}: {reason}" for backend, reason in unavailable.items())
        raise RuntimeError(f"requested backend unavailable: {details}")
    selected = [backend for backend in requested if backend not in unavailable]
    skipped = [backend for backend in requested if backend in unavailable]
    return selected, skipped


def resolved_scenario(args: argparse.Namespace, config: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(scenario)
    if "request_rate_fraction" in resolved:
        capacity_rps = args.capacity_rps if args.capacity_rps is not None else config.get("capacity_rps")
        if capacity_rps is None or float(capacity_rps) <= 0.0:
            raise RuntimeError("request_rate_fraction requires a positive --capacity-rps or config capacity_rps")
        resolved["request_rate_rps"] = float(resolved["request_rate_fraction"]) * float(capacity_rps)
        resolved["capacity_rps"] = float(capacity_rps)
    return resolved


def command_output(command: List[str]) -> str:
    try:
        run = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return ""
    return run.stdout.strip() if run.returncode == 0 else ""


def python_environment(executable: str) -> Dict[str, Any]:
    script = (
        "import importlib.metadata as m\n"
        "import json,platform,sys\n"
        "versions={}\n"
        "for name in ('torch','transformers','vllm'):\n"
        "    try:\n"
        "        versions[name]=m.version(name)\n"
        "    except m.PackageNotFoundError:\n"
        "        pass\n"
        "print(json.dumps({'executable':sys.executable,'python':platform.python_version(),'packages':versions}))\n"
    )
    try:
        run = subprocess.run([executable, "-c", script], check=False, capture_output=True, text=True)
    except OSError as exc:
        return {"executable": executable, "error": str(exc)}
    if run.returncode != 0:
        return {"executable": executable, "error": run.stderr.strip()}
    return json.loads(run.stdout)


def collect_environment(vllm_python: str) -> Dict[str, Any]:
    git_commit = command_output(["git", "rev-parse", "HEAD"])
    git_status = command_output(["git", "status", "--porcelain=v1"])
    gpu_csv = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    snapshots = {"suite": python_environment(sys.executable)}
    # A venv may symlink the same interpreter binary while exposing a different
    # package environment, so compare the requested executable paths rather
    # than collapsing them with realpath().
    if os.path.abspath(vllm_python) != os.path.abspath(sys.executable):
        snapshots["vllm"] = python_environment(vllm_python)
    return {
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "git_status": git_status.splitlines(),
        "gpu_driver": gpu_csv.splitlines(),
        "python": snapshots,
    }


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
    traffic_mode = str(scenario.get("traffic_mode", defaults.get("traffic_mode", "offline")))
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
        "--traffic-mode",
        traffic_mode,
        "--tinyllm-dtype",
        str(scenario.get("tinyllm_dtype", defaults.get("tinyllm_dtype", "fp32"))),
        "--tinyllm-kv-cache-dtype",
        str(scenario.get("tinyllm_kv_cache_dtype", defaults.get("tinyllm_kv_cache_dtype", "fp32"))),
        "--vllm-python",
        str(args.vllm_python or config.get("vllm_python", sys.executable)),
        "--vllm-dtype",
        str(scenario.get("vllm_dtype", defaults.get("vllm_dtype", "float32"))),
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
    vllm_python = str(args.vllm_python or config.get("vllm_python", sys.executable))

    report: Dict[str, Any] = {
        "label": label,
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "model_dir": str(model_dir),
        "device": str(args.device or config.get("device", "cuda:0")),
        "invocation": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "environment": collect_environment(vllm_python),
        "scenarios": [],
    }

    for configured_scenario in config.get("workloads", []):
        scenario = resolved_scenario(args, config, configured_scenario)
        workload_path = workload_dir / f"{scenario['name']}.jsonl"
        workload_meta = (
            {"path": str(workload_path), "dry_run": True}
            if args.dry_run
            else write_workload_jsonl(workload_path, tokenizer, scenario, dict(config.get("defaults", {})))
        )
        backends, skipped = resolve_backends(args, config, scenario)
        events_path = events_dir / f"{scenario['name']}.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        scenario_report: Dict[str, Any] = {
            "workload": scenario,
            "workload_jsonl": str(workload_path),
            "workload_meta": workload_meta,
            "benchmark_mode": str(scenario.get("benchmark_mode", config.get("defaults", {}).get("benchmark_mode", "-"))),
            "traffic_mode": str(scenario.get("traffic_mode", config.get("defaults", {}).get("traffic_mode", "offline"))),
            "command": [],
            "selected_backends": backends,
            "skipped_backends": skipped,
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
            scenario_report["output_agreement"] = comparison.get("output_agreement", {})
            event_metrics = read_event_metrics(events_path)
            scenario_report["tinyllm_events"] = event_metrics
            if "tinyllm" in backends and not args.dry_run and not event_metrics.get("complete", False):
                raise RuntimeError(
                    "TinyLLM event trace is incomplete: " + "; ".join(event_metrics.get("completeness_errors", []))
                )
            agreement = scenario_report["output_agreement"].get("tinyllm_vs_transformers", {})
            if scenario_report["benchmark_mode"] == "correctness" and agreement and not agreement.get("match", False):
                raise RuntimeError("TinyLLM and Transformers greedy token IDs differ")
        except Exception as exc:
            scenario_report["error"] = str(exc)
        report["scenarios"].append(scenario_report)

    paths = write_reports(run_dir, label, report)
    print(f"wrote JSON report: {paths['json']}")
    print(f"wrote Markdown report: {paths['markdown']}")
    return 1 if any(scenario.get("error") for scenario in report["scenarios"]) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"run_benchmark_suite failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
