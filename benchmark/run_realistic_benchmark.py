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
    parser.add_argument("--artifact-checksum-file")
    parser.add_argument("--phase", choices=("trace", "offline", "publish", "all"), default="all")
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


def public_file_record(path: Path) -> Dict[str, Any]:
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_public_manifest(
    prepared_manifest: Dict[str, Any], config_sources: Dict[str, Any], model_dir: Path
) -> Dict[str, Any]:
    manifest = dict(prepared_manifest)
    manifest["sources"] = config_sources
    for source_name in ("burstgpt", "oasst1"):
        source = dict(config_sources[source_name])
        prepared = dict(manifest.get(source_name, {}))
        path_text = prepared.pop("path", "")
        if path_text:
            source_path = Path(path_text)
            if not source_path.is_file():
                raise RuntimeError(f"missing {source_name} source file: {source_path}")
            prepared.update(public_file_record(source_path))
        actual_sha256 = str(prepared.get("sha256", ""))
        expected_sha256 = str(source.get("sha256", ""))
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{source_name} SHA-256 mismatch during publish: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        manifest[source_name] = {**source, **prepared}

    required_model_files = [model_dir / "config.json", model_dir / "tokenizer.json"]
    optional_model_files = [model_dir / "tokenizer_config.json", model_dir / "generation_config.json"]
    model_files = required_model_files + [path for path in optional_model_files if path.is_file()]
    model_files.extend(sorted(model_dir.glob("*.safetensors")))
    missing_model_files = [str(path) for path in required_model_files if not path.is_file()]
    if missing_model_files or not any(path.suffix == ".safetensors" for path in model_files):
        details = ", ".join(missing_model_files) or "no safetensors files"
        raise RuntimeError(f"incomplete model files during publish: {details}")
    manifest["model"] = {
        **config_sources["model"],
        "local_path": str(model_dir),
        "files": [public_file_record(path) for path in model_files],
    }
    manifest.pop("model_dir", None)
    return manifest


def range_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def refresh_trace_summaries(trace: Dict[str, Any], slo_config: Dict[str, Any]) -> None:
    for window in trace.get("windows", {}).values():
        for load in window.get("loads", {}).values():
            requests = load.get("request_metrics", {}).get("requests", [])
            load["request_metrics"] = enrich_and_group_requests(requests, requests)
        reference_key = f"{float(slo_config['reference_load_fraction']):.2f}C"
        reference = window.get("loads", {}).get(reference_key)
        if not reference:
            continue
        metrics = reference["request_metrics"]["overall"]["metrics"]
        ttft_slo = float(metrics["ttft_ms"]["p99"]) * float(slo_config["ttft_p99_multiplier"])
        tpot_slo = float(metrics["tpot_ms"]["p99"]) * float(slo_config["tpot_p99_multiplier"])
        for load in window["loads"].values():
            load["relative_goodput"] = relative_goodput(
                load["request_metrics"]["requests"], ttft_slo, tpot_slo
            )

    aggregates: Dict[str, Any] = {}
    load_names = sorted(
        {name for window in trace.get("windows", {}).values() for name in window.get("loads", {})}
    )
    for load_name in load_names:
        loads = [
            window["loads"][load_name]
            for window in trace.get("windows", {}).values()
            if load_name in window.get("loads", {})
        ]
        scalar_getters = {
            "target_rps": lambda load: float(load["target_rps"]),
            "achieved_request_per_s": lambda load: float(load["request_metrics"]["overall"]["request_per_s"]),
            "input_tokens_per_s": lambda load: float(load["request_metrics"]["overall"]["input_tokens_per_s"]),
            "output_tokens_per_s": lambda load: float(load["request_metrics"]["overall"]["output_tokens_per_s"]),
            "total_tokens_per_s": lambda load: float(load["request_metrics"]["overall"]["total_tokens_per_s"]),
            "wall_clock_ms": lambda load: float(load["request_metrics"]["overall"]["wall_clock_ms"]),
            "max_concurrency": lambda load: float(load["request_metrics"]["overall"]["max_concurrency"]),
            "good_request_ratio": lambda load: float(load["relative_goodput"]["good_request_ratio"]),
        }
        for metric_name in ("queue_ms", "ttft_ms", "engine_ttft_ms", "tpot_ms", "e2e_ms"):
            for percentile_name in ("p50", "p95", "p99"):
                scalar_getters[f"{metric_name}_{percentile_name}"] = (
                    lambda load, metric_name=metric_name, percentile_name=percentile_name: float(
                        load["request_metrics"]["overall"]["metrics"][metric_name][percentile_name]
                    )
                )
        aggregates[load_name] = {
            name: range_summary([getter(load) for load in loads]) for name, getter in scalar_getters.items()
        }
    trace["cross_window_summary"] = aggregates


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
    environment = manifest.get("environment", {})
    lines = [
        "# TinyLLM Realistic Workload Benchmark v1",
        "",
        "## Executive summary",
        "",
        "This benchmark combines BurstGPT arrival/length traces with length-matched OASST1 prompts.",
        "It is a single-process, single-device internal submission benchmark, not a production HTTP SLA.",
        "",
        "The trace replay uses three non-overlapping 1,000-request windows and calibrates each window",
        "against its own simultaneous saturation throughput before replaying 0.25C through 0.90C.",
        "The offline section compares identical prompt cohorts across TinyLLM, Transformers, and vLLM.",
        "",
        "## Scope and claim boundary",
        "",
        "Results apply only to the recorded commit, model, GPU, dtype, and workloads. Relative goodput",
        "uses experiment-local thresholds and is not a production SLA or a claim of serving parity.",
        "",
        "## Environment",
        "",
        f"- Runtime candidate commit: `{environment.get('git_commit', '-')}`",
        f"- git_dirty: `{environment.get('git_dirty', '-')}`",
        f"- GPU / driver: `{'; '.join(environment.get('gpu_driver', [])) or '-'}`",
        f"- CUDA toolkit: `{manifest.get('cuda_toolkit', '-')}`",
        "- TinyLLM compute / KV dtype: `FP32 / FP32`",
        "- Sampling: greedy; correctness retains EOS, performance uses fixed output",
        "",
        "## Sources and workload construction",
        "",
        "- OASST1 is a content proxy; it is not the original BurstGPT request text.",
        "- BurstGPT token counts come from GPT services and are matched approximately under the Qwen tokenizer.",
        "- Arrival timestamps are linearly scaled to workload-specific measured capacity.",
        "- Fixed output lengths model requested inference work and ignore natural EOS behavior.",
        "- Network, HTTP, authentication, multi-GPU, and production reliability overheads are excluded.",
        "",
        "## Dataset manifest",
        "",
        f"- BurstGPT revision: `{manifest.get('burstgpt', {}).get('revision', '-')}`",
        f"- BurstGPT SHA-256: `{manifest.get('burstgpt', {}).get('sha256', '-')}`",
        f"- OASST1 revision: `{manifest.get('oasst1', {}).get('revision', '-')}`",
        f"- OASST1 SHA-256: `{manifest.get('oasst1', {}).get('sha256', '-')}`",
        f"- Qwen revision: `{manifest.get('model', {}).get('revision', '-')}`",
        f"- Windows: `{manifest.get('window_count', '-')}` × `{manifest.get('window_size', '-')}` requests",
        "",
        "## Trace capacity and replay results",
        "",
        "`C` is measured independently for each window as `1000 × 1000 / generation_ms`.",
        "",
        "| Window | Capacity req/s | Load | Target req/s | Completed | Req/s | TTFT p99 ms | TPOT p99 ms | E2E p99 ms | Max conc. | Good ratio |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for window_id, window in sorted(trace.get("windows", {}).items()):
        for load_name, load in sorted(window.get("loads", {}).items()):
            summary = load["request_metrics"]["overall"]
            metrics = summary["metrics"]
            lines.append(
                f"| {window_id} | {window['capacity']['capacity_rps']:.6f} | {load_name} | "
                f"{load['target_rps']:.6f} | {summary['completed_request_count']}/{summary['request_count']} | "
                f"{summary['request_per_s']:.6f} | {metrics['ttft_ms']['p99']:.3f} | "
                f"{metrics['tpot_ms']['p99']:.3f} | {metrics['e2e_ms']['p99']:.3f} | "
                f"{summary['max_concurrency']} | "
                f"{load['relative_goodput']['good_request_ratio']:.4f} |"
            )
    lines.extend(
        [
            "",
            "The JSON report additionally contains p50/p95/p99 queue, TTFT, engine TTFT, TPOT, and E2E",
            "for the overall population and every log-type, ISL, and OSL bucket, together with input/output/total",
            "token throughput, wall-clock duration, and cross-window min/median/max summaries.",
            "",
            "## Offline three-backend cohorts",
            "",
        ]
    )
    for cohort, aggregate in sorted(offline.get("aggregates", {}).items()):
        lines.extend([f"### {cohort}", "", "| Backend | E2E tok/s median | Decode tok/s median | Latency ms median |", "| --- | ---: | ---: | ---: |"])
        for backend, metrics in sorted(aggregate["backends"].items()):
            lines.append(
                f"| {backend} | {metrics['end_to_end_tokens_per_s']['p50']:.3f} | "
                f"{metrics['decode_tokens_per_s']['p50']:.3f} | {metrics['avg_total_latency_ms']['p50']:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Relative goodput",
            "",
            "For each window, `TTFT SLO = 2.0 × that window's 0.25C TTFT p99` and",
            "`TPOT SLO = 1.5 × that window's 0.25C TPOT p99`. A request counts as good only if both",
            "conditions hold. These are relative experiment thresholds, not production objectives.",
            "",
            "## Reproduction",
            "",
            "See `manifest.json` for the exact source revisions, file hashes, environment, configuration,",
            "and complete preparation and execution commands. Real prompt text remains only in the ignored",
            "server-side workload artifacts; `selection.json` contains source metadata and prompt hashes.",
            "",
            "## Raw artifacts",
            "",
            "The server-side archive contains prepared private workloads, capacity and replay workloads,",
            "request events, and unsanitized run JSON. Its SHA-256 is recorded in `manifest.json`; the archive",
            "is intentionally not committed to Git.",
        ]
    )
    return "\n".join(lines) + "\n"


def publish(args: argparse.Namespace, prepared: Path, output: Path, trace: Dict[str, Any], offline: Dict[str, Any]) -> None:
    if not args.publish_dir:
        return
    publish_dir = Path(args.publish_dir)
    prepared_manifest = json.loads((prepared / "workload-manifest.json").read_text(encoding="utf-8"))
    manifest = build_public_manifest(
        prepared_manifest, args.config_data["sources"], Path(args.model_dir)
    )
    selection = json.loads((prepared / "selection.json").read_text(encoding="utf-8"))
    refresh_trace_summaries(trace, args.config_data["relative_slo"])
    trace_environment = trace.get("environment", {})
    offline_environment = offline.get("environment", {})
    trace_commit = str(trace_environment.get("git_commit", ""))
    offline_commit = str(offline_environment.get("git_commit", ""))
    if trace_commit != offline_commit:
        raise RuntimeError(f"trace/offline candidate commit mismatch: {trace_commit} != {offline_commit}")
    if trace_environment.get("git_dirty") or offline_environment.get("git_dirty"):
        raise RuntimeError("refusing to publish results produced from a dirty worktree")
    nvcc = subprocess.run(
        ["/usr/local/cuda-12.8/bin/nvcc", "--version"], check=False, capture_output=True, text=True
    )
    manifest["environment"] = trace_environment
    manifest["cuda_toolkit"] = nvcc.stdout.strip().splitlines()[-1] if nvcc.returncode == 0 else "unknown"
    manifest["report_generated_by_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    ).stdout.strip()
    manifest["configuration"] = args.config_data
    manifest["commands"] = {
        "prepare": (
            "python3 benchmark/prepare_realistic_workload.py "
            "--config benchmark/configs/qwen25_realistic_v1.json "
            "--burstgpt <dataset-root>/raw/burstgpt/BurstGPT_without_fails_3.csv "
            "--oasst <dataset-root>/raw/oasst1/2023-04-12_oasst_ready.trees.jsonl.gz "
            "--model-dir /models/Qwen2.5-1.5B-Instruct --output-dir <dataset-root>/generated"
        ),
        "run": (
            "python3 benchmark/run_realistic_benchmark.py --prepared-dir <dataset-root>/generated "
            "--config benchmark/configs/qwen25_realistic_v1.json "
            "--model-dir /models/Qwen2.5-1.5B-Instruct "
            "--tinyllm-binary build/cuda-release/benchmark/llama_engine_benchmark "
            "--vllm-python /root/autodl-tmp/venvs/vllm/bin/python "
            "--output-dir <artifact-root>/run --publish-dir benchmark/reports/realistic-v1 --phase all --resume"
        ),
    }
    if args.artifact_checksum_file:
        checksum_text = Path(args.artifact_checksum_file).read_text(encoding="utf-8").strip()
        manifest["raw_artifact"] = {
            "filename": Path(checksum_text.split()[-1].lstrip("*")).name,
            "sha256": checksum_text.split()[0],
        }
    publish_dir.mkdir(parents=True, exist_ok=True)
    (publish_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_json(publish_dir / "trace_replay.json", trace)
    save_json(publish_dir / "offline.json", offline)
    (publish_dir / "README.md").write_text(build_markdown(trace, offline, manifest), encoding="utf-8")
    manifest["published_files"] = {
        name: {"bytes": (publish_dir / name).stat().st_size, "sha256": sha256_file(publish_dir / name)}
        for name in ("README.md", "selection.json", "trace_replay.json", "offline.json")
    }
    save_json(publish_dir / "manifest.json", manifest)


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
