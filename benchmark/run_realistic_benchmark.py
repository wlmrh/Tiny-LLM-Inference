#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from run_benchmark_suite import collect_environment, parse_json_line
from suite.metrics import enrich_and_group_requests, read_event_metrics, relative_goodput, summarize_values
from suite.realistic import (
    build_offline_cohorts,
    public_selection,
    read_workload_jsonl,
    scale_trace_arrivals,
    sha256_file,
    write_workload_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TinyLLM realistic benchmark v1")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--config", default="benchmark/configs/qwen25_realistic_v1.json")
    parser.add_argument("--model-dir", default="/models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tinyllm-binary", default="build/cuda-release/benchmark/llama_engine_benchmark")
    parser.add_argument("--vllm-python", default="/root/autodl-tmp/venvs/vllm/bin/python")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--publish-dir")
    parser.add_argument("--phase", choices=("trace", "offline", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.config_data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    return args


def run_json(command: List[str]) -> Dict[str, Any]:
    run = subprocess.run(command, check=False, capture_output=True, text=True)
    if run.returncode != 0:
        sys.stderr.write(run.stdout)
        sys.stderr.write(run.stderr)
        raise RuntimeError(f"command failed with exit code {run.returncode}: {' '.join(command)}")
    return parse_json_line(run.stdout)


def sanitize_result(data: Any) -> Any:
    if isinstance(data, list):
        return [sanitize_result(item) for item in data]
    if not isinstance(data, dict):
        return data
    return {
        key: sanitize_result(value)
        for key, value in data.items()
        if key not in {"samples", "request_metrics", "prompt", "output_text", "generated_text"}
    }


def tiny_command(args: argparse.Namespace, workload: Path, events: Path) -> List[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve().with_name("run_benchmark_comparison.py")),
        "--tinyllm-binary",
        args.tinyllm_binary,
        "--backend",
        "tinyllm",
        "--device",
        "cuda:0",
        "--tinyllm-dtype",
        "fp32",
        "--tinyllm-kv-cache-dtype",
        "fp32",
        "--warmup",
        "1",
        "--warmup-request-count",
        str(int(args.config_data["warmup_request_count"])),
        "--repeat",
        "1",
        "--max-new-tokens",
        "512",
        "--ignore-eos",
        "--traffic-mode",
        "open-loop",
        "--benchmark-mode",
        "fixed_output_perf",
        "--workload-jsonl",
        str(workload),
        "--events-jsonl",
        str(events),
        "--json",
        args.model_dir,
    ]


def comparison_command(
    args: argparse.Namespace, workload: Path, events: Path, output_tokens: int, correctness: bool
) -> List[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("run_benchmark_comparison.py")),
        "--tinyllm-binary",
        args.tinyllm_binary,
        "--vllm-python",
        args.vllm_python,
        "--backend",
        "tinyllm,transformers,vllm",
        "--device",
        "cuda:0",
        "--tinyllm-dtype",
        "fp32",
        "--tinyllm-kv-cache-dtype",
        "fp32",
        "--vllm-dtype",
        "float32",
        "--warmup",
        "0" if correctness else "1",
        "--repeat",
        "1",
        "--max-new-tokens",
        str(output_tokens),
        "--benchmark-mode",
        "correctness" if correctness else "fixed_output_perf",
        "--workload-jsonl",
        str(workload),
        "--events-jsonl",
        str(events),
        "--json",
    ]
    if not correctness:
        command.append("--ignore-eos")
    command.append(args.model_dir)
    return command


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_or_initialize(path: Path, kind: str, environment: Dict[str, Any]) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "benchmark": kind,
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "environment": environment,
        "windows": {},
        "scenarios": [],
    }


def run_trace(args: argparse.Namespace, prepared: Path, output: Path, environment: Dict[str, Any]) -> Dict[str, Any]:
    state_path = output / "trace_replay.json"
    state = load_or_initialize(state_path, "realistic_trace_replay", environment)
    for base_path in sorted((prepared / "base").glob("window-*.jsonl")):
        window_id = base_path.stem
        base_records = read_workload_jsonl(base_path)
        window = state["windows"].setdefault(window_id, {"capacity": None, "loads": {}})
        if window["capacity"] is None:
            capacity_records = [dict(item, arrival_ms=0.0) for item in base_records]
            workload = output / "workloads" / f"{window_id}-capacity.jsonl"
            events = output / "events" / f"{window_id}-capacity.jsonl"
            write_workload_jsonl(workload, capacity_records)
            comparison = run_json(tiny_command(args, workload, events))
            result = comparison["results"][0]
            event_metrics = read_event_metrics(events)
            if not event_metrics.get("complete") or len(event_metrics.get("requests", [])) != len(base_records):
                raise RuntimeError(f"incomplete capacity event stream for {window_id}")
            capacity_rps = len(base_records) * 1000.0 / float(result["avg_total_latency_ms"])
            window["capacity"] = {
                "requests": len(base_records),
                "capacity_rps": capacity_rps,
                "result": sanitize_result(result),
                "events_sha256": sha256_file(events),
            }
            save_json(state_path, state)
        capacity_rps = float(window["capacity"]["capacity_rps"])
        for fraction in (float(value) for value in args.config_data["load_fractions"]):
            key = f"{fraction:.2f}C"
            if args.resume and key in window["loads"]:
                continue
            target_rps = capacity_rps * fraction
            records = scale_trace_arrivals(base_records, target_rps)
            workload = output / "workloads" / f"{window_id}-{fraction:.2f}c.jsonl"
            events = output / "events" / f"{window_id}-{fraction:.2f}c.jsonl"
            write_workload_jsonl(workload, records)
            comparison = run_json(tiny_command(args, workload, events))
            event_metrics = read_event_metrics(events)
            if not event_metrics.get("complete") or len(event_metrics.get("requests", [])) != len(records):
                raise RuntimeError(f"incomplete trace event stream for {window_id} {key}")
            grouped = enrich_and_group_requests(event_metrics["requests"], records)
            planned_rate = (len(records) - 1) * 1000.0 / float(records[-1]["arrival_ms"])
            window["loads"][key] = {
                "fraction": fraction,
                "target_rps": target_rps,
                "planned_rps": planned_rate,
                "offered_load_error_ratio": abs(planned_rate - target_rps) / target_rps,
                "result": sanitize_result(comparison["results"][0]),
                "request_metrics": grouped,
                "events_sha256": sha256_file(events),
            }
            save_json(state_path, state)
        slo_config = args.config_data["relative_slo"]
        reference_key = f"{float(slo_config['reference_load_fraction']):.2f}C"
        reference = window["loads"][reference_key]["request_metrics"]
        ttft_slo = float(reference["overall"]["metrics"]["ttft_ms"]["p99"]) * float(
            slo_config["ttft_p99_multiplier"]
        )
        tpot_slo = float(reference["overall"]["metrics"]["tpot_ms"]["p99"]) * float(
            slo_config["tpot_p99_multiplier"]
        )
        for load in window["loads"].values():
            load["relative_goodput"] = relative_goodput(
                load["request_metrics"]["requests"], ttft_slo, tpot_slo
            )
        save_json(state_path, state)
    return state


def aggregate_offline(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregates: Dict[str, Any] = {}
    for cohort in sorted({str(item["cohort"]) for item in scenarios}):
        cohort_items = [item for item in scenarios if item["cohort"] == cohort]
        backend_names = sorted(
            {str(result["backend"]) for item in cohort_items for result in item["comparison"].get("results", [])}
        )
        backend_summary: Dict[str, Any] = {}
        for backend in backend_names:
            results = [
                result
                for item in cohort_items
                for result in item["comparison"].get("results", [])
                if result["backend"] == backend
            ]
            backend_summary[backend] = {
                metric: summarize_values([float(result.get(metric, 0.0)) for result in results])
                for metric in (
                    "avg_total_latency_ms",
                    "avg_first_token_latency_ms",
                    "end_to_end_tokens_per_s",
                    "decode_tokens_per_s",
                )
            }
        aggregates[cohort] = {"shards": len(cohort_items), "backends": backend_summary}
    return aggregates


def run_offline(args: argparse.Namespace, prepared: Path, output: Path, environment: Dict[str, Any]) -> Dict[str, Any]:
    state_path = output / "offline.json"
    state = load_or_initialize(state_path, "realistic_offline_comparison", environment)
    all_records = [
        record for path in sorted((prepared / "base").glob("window-*.jsonl")) for record in read_workload_jsonl(path)
    ]
    cohorts = build_offline_cohorts(all_records)
    completed = {(item["cohort"], int(item["shard"])) for item in state["scenarios"]}
    for cohort_name, shards in cohorts.items():
        for shard_index, records in enumerate(shards, start=1):
            if args.resume and (cohort_name, shard_index) in completed:
                continue
            workload = output / "offline-workloads" / f"{cohort_name}-{shard_index:02d}.jsonl"
            events = output / "offline-events" / f"{cohort_name}-{shard_index:02d}.jsonl"
            write_workload_jsonl(workload, records)
            output_tokens = int(records[0]["max_new_tokens"])
            comparison = run_json(
                comparison_command(args, workload, events, output_tokens, cohort_name == "correctness")
            )
            if cohort_name == "correctness":
                for name, agreement in comparison.get("output_agreement", {}).items():
                    if not agreement.get("match", False):
                        raise RuntimeError(f"correctness mismatch for {name}")
            if cohort_name != "correctness" and comparison.get("ratio_status") and any(
                not item.get("valid", False) for item in comparison["ratio_status"].values()
            ):
                raise RuntimeError(f"generated token mismatch in {cohort_name} shard {shard_index}")
            state["scenarios"].append(
                {
                    "cohort": cohort_name,
                    "shard": shard_index,
                    "request_count": len(records),
                    "selection": public_selection(records),
                    "comparison": sanitize_result(comparison),
                }
            )
            state["aggregates"] = aggregate_offline(state["scenarios"])
            save_json(state_path, state)
    return state


def build_markdown(trace: Dict[str, Any], offline: Dict[str, Any], manifest: Dict[str, Any]) -> str:
    lines = [
        "# TinyLLM Realistic Workload Benchmark v1",
        "",
        "This benchmark combines BurstGPT arrival/length traces with length-matched OASST1 prompts.",
        "It is a single-process, single-device internal submission benchmark, not a production HTTP SLA.",
        "",
        "## Sources and limitations",
        "",
        "- OASST1 is a content proxy; it is not the original BurstGPT request text.",
        "- BurstGPT token counts come from GPT services and are matched approximately under the Qwen tokenizer.",
        "- Arrival timestamps are linearly scaled to workload-specific measured capacity.",
        "- Fixed output lengths model requested inference work and ignore natural EOS behavior.",
        "- Network, HTTP, authentication, multi-GPU, and production reliability overheads are excluded.",
        "",
        "## Dataset manifest",
        "",
        f"- BurstGPT SHA-256: `{manifest.get('burstgpt', {}).get('sha256', '-')}`",
        f"- OASST1 SHA-256: `{manifest.get('oasst1', {}).get('sha256', '-')}`",
        f"- Windows: `{manifest.get('window_count', '-')}` × `{manifest.get('window_size', '-')}` requests",
        "",
        "## Trace replay",
        "",
        "| Window | Capacity req/s | Load | Target req/s | TTFT p99 ms | TPOT p99 ms | E2E p99 ms | Good ratio |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for window_id, window in sorted(trace.get("windows", {}).items()):
        for load_name, load in sorted(window.get("loads", {}).items()):
            overall = load["request_metrics"]["overall"]["metrics"]
            lines.append(
                f"| {window_id} | {window['capacity']['capacity_rps']:.6f} | {load_name} | "
                f"{load['target_rps']:.6f} | {overall['ttft_ms']['p99']:.3f} | "
                f"{overall['tpot_ms']['p99']:.3f} | {overall['e2e_ms']['p99']:.3f} | "
                f"{load['relative_goodput']['good_request_ratio']:.4f} |"
            )
    lines.extend(["", "## Offline three-backend cohorts", ""])
    for cohort, aggregate in sorted(offline.get("aggregates", {}).items()):
        lines.extend([f"### {cohort}", "", "| Backend | E2E tok/s median | Decode tok/s median | Latency ms median |", "| --- | ---: | ---: | ---: |"])
        for backend, metrics in sorted(aggregate["backends"].items()):
            lines.append(
                f"| {backend} | {metrics['end_to_end_tokens_per_s']['p50']:.3f} | "
                f"{metrics['decode_tokens_per_s']['p50']:.3f} | {metrics['avg_total_latency_ms']['p50']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def publish(args: argparse.Namespace, prepared: Path, output: Path, trace: Dict[str, Any], offline: Dict[str, Any]) -> None:
    if not args.publish_dir:
        return
    publish_dir = Path(args.publish_dir)
    manifest = json.loads((prepared / "workload-manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((prepared / "selection.json").read_text(encoding="utf-8"))
    publish_dir.mkdir(parents=True, exist_ok=True)
    save_json(publish_dir / "manifest.json", manifest)
    (publish_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_json(publish_dir / "trace_replay.json", trace)
    save_json(publish_dir / "offline.json", offline)
    (publish_dir / "README.md").write_text(build_markdown(trace, offline, manifest), encoding="utf-8")


def main() -> int:
    args = parse_args()
    prepared = Path(args.prepared_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(args.vllm_python)
    trace = json.loads((output / "trace_replay.json").read_text()) if (output / "trace_replay.json").exists() else {}
    offline = json.loads((output / "offline.json").read_text()) if (output / "offline.json").exists() else {}
    if args.phase in {"trace", "all"}:
        trace = run_trace(args, prepared, output, environment)
    if args.phase in {"offline", "all"}:
        offline = run_offline(args, prepared, output, environment)
    if trace and offline:
        publish(args, prepared, output, trace, offline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
