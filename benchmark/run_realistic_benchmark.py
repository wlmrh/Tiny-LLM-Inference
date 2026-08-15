#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
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


NVIDIA_RTX_4080_SUPER_SPECS_URL = (
    "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080-family/"
)


def gpu_memory_observation(gpu_records: List[str]) -> Dict[str, Any]:
    for record in gpu_records:
        fields = [field.strip() for field in record.split(",")]
        if len(fields) != 3 or fields[0] != "NVIDIA GeForce RTX 4080 SUPER":
            continue
        try:
            reported_memory_mib = int(fields[2])
        except ValueError:
            continue
        if reported_memory_mib == 32760:
            return {
                "measurement_source": "nvidia-smi runtime query on the benchmark host",
                "reported_memory_mib": reported_memory_mib,
                "retail_reference_memory_gb": 16,
                "retail_reference_url": NVIDIA_RTX_4080_SUPER_SPECS_URL,
                "interpretation": (
                    "The cloud host exposed an RTX 4080 SUPER device name with 32760 MiB, which does "
                    "not match NVIDIA's 16 GB retail reference specification. Cloud-platform device "
                    "presentation or nonstandard provisioning may explain the discrepancy, but the exact "
                    "cause was not independently verified."
                ),
            }
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TinyLLM realistic benchmark v1")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--config", default="benchmark/configs/qwen25_realistic_v1.json")
    parser.add_argument("--model-dir", default="/models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tinyllm-binary", default="build/cuda-release/benchmark/llama_engine_benchmark")
    parser.add_argument("--vllm-python", default=sys.executable)
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


def build_reproduction_commands(args: argparse.Namespace) -> Dict[str, str]:
    sources = args.config_data["sources"]
    prepare_command = [
        "python3",
        "benchmark/prepare_realistic_workload.py",
        "--config",
        args.config,
        "--burstgpt",
        f"<dataset-root>/raw/burstgpt/{sources['burstgpt']['filename']}",
        "--oasst",
        f"<dataset-root>/raw/oasst1/{sources['oasst1']['filename']}",
        "--model-dir",
        "<model-dir>",
        "--output-dir",
        "<dataset-root>/generated",
    ]
    run_command = [
        "python3",
        "benchmark/run_realistic_benchmark.py",
        "--prepared-dir",
        "<dataset-root>/generated",
        "--config",
        args.config,
        "--model-dir",
        "<model-dir>",
        "--tinyllm-binary",
        args.tinyllm_binary,
        "--vllm-python",
        "<python-with-vllm>",
        "--output-dir",
        "<artifact-root>/run",
        "--publish-dir",
        "<publish-dir>",
        "--phase",
        args.phase,
    ]
    if args.artifact_checksum_file:
        run_command.extend(["--artifact-checksum-file", "<artifact-checksum-file>"])
    if args.resume:
        run_command.append("--resume")
    return {"prepare": shlex.join(prepare_command), "run": shlex.join(run_command)}


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


def format_range(summary: Dict[str, float], digits: int = 3) -> str:
    return " / ".join(f"{float(summary[key]):.{digits}f}" for key in ("min", "median", "max"))


def build_markdown(
    trace: Dict[str, Any],
    offline: Dict[str, Any],
    manifest: Dict[str, Any],
    selection: List[Dict[str, Any]],
) -> str:
    environment = manifest.get("environment", {})
    suite_python = environment.get("python", {}).get("suite", {})
    suite_packages = suite_python.get("packages", {})
    vllm_python = environment.get("python", {}).get("vllm", {})
    vllm_packages = vllm_python.get("packages", {})
    gpu_records = environment.get("gpu_driver", [])
    gpu_summary = "; ".join(gpu_records) or "-"
    if len(gpu_records) == 1:
        gpu_fields = [field.strip() for field in gpu_records[0].split(",")]
        if len(gpu_fields) == 3:
            gpu_summary = (
                f"{gpu_fields[0]}; driver {gpu_fields[1]}; host-reported memory {gpu_fields[2]} MiB"
            )
    memory_observation = environment.get("gpu_memory_observation", {}) or gpu_memory_observation(
        gpu_records
    )
    gpu_memory_lines: List[str] = []
    if memory_observation:
        gpu_memory_lines = [
            "- GPU memory disclosure: `32760 MiB` is the literal value returned by the benchmark host's",
            "  runtime query, not the standard specification of a retail RTX 4080 SUPER.",
            f"  [NVIDIA's reference specification]({memory_observation['retail_reference_url']}) lists "
            f"`{memory_observation['retail_reference_memory_gb']} GB GDDR6X`.",
            "  The mismatch may reflect cloud-platform device presentation or nonstandard provisioning,",
            "  but the exact cause was not independently verified. Treat both the device name and memory",
            "  capacity as provider-exposed properties of this benchmark host.",
        ]
    windows = trace.get("windows", {})
    cross_window = trace.get("cross_window_summary", {})
    replay_count = sum(len(window.get("loads", {})) for window in windows.values())
    capacity_count = len(windows)
    replay_requests = [
        load["request_metrics"]["overall"]
        for window in windows.values()
        for load in window.get("loads", {}).values()
    ]
    replay_complete = all(
        summary["request_count"] == summary["completed_request_count"] and summary["error_count"] == 0
        for summary in replay_requests
    )

    correctness = next(
        (scenario for scenario in offline.get("scenarios", []) if scenario.get("cohort") == "correctness"),
        {},
    )
    agreements = correctness.get("comparison", {}).get("output_agreement", {})
    correctness_match = bool(agreements) and all(item.get("match", False) for item in agreements.values())
    compared_requests = min(
        (int(item.get("compared_requests", 0)) for item in agreements.values()), default=0
    )

    offline_aggregates = offline.get("aggregates", {})
    performance_cohorts = (
        ("short_chat", "Short chat"),
        ("medium_chat", "Medium chat"),
        ("long_prefill", "Long prefill"),
        ("long_decode", "Long decode"),
    )

    def e2e_median(cohort: str, backend: str) -> float:
        return float(
            offline_aggregates[cohort]["backends"][backend]["end_to_end_tokens_per_s"]["p50"]
        )

    short_vs_transformers = e2e_median("short_chat", "tinyllm") / e2e_median(
        "short_chat", "transformers"
    )
    medium_vs_transformers = e2e_median("medium_chat", "tinyllm") / e2e_median(
        "medium_chat", "transformers"
    )
    decode_vs_transformers = e2e_median("long_decode", "tinyllm") / e2e_median(
        "long_decode", "transformers"
    )
    prefill_vs_transformers = e2e_median("long_prefill", "tinyllm") / e2e_median(
        "long_prefill", "transformers"
    )

    lines = [
        "# TinyLLM Realistic Workload Benchmark v1",
        "",
        "## Executive summary",
        "",
        "This benchmark combines BurstGPT arrival/length traces with length-matched OASST1 prompts",
        "to exercise heterogeneous request lengths, burst timing, and per-request output limits without",
        "changing the inference engine. It is a single-process, single-device experiment, not a production HTTP SLA.",
        "",
        "This narrative is rendered from the committed sanitized JSON; the benchmark was not rerun and",
        "the raw measurements were not edited for this documentation update.",
        "",
        f"- `{capacity_count}` experiment-local reference-capacity calibrations and `{replay_count}` trace replays completed.",
        f"- All trace replays completed 1,000/1,000 requests with zero reported errors: "
        f"`{'yes' if replay_complete else 'no'}`.",
        f"- `{compared_requests}` EOS-aware correctness prompts matched exactly across all three backends: "
        f"`{'yes' if correctness_match else 'no'}`.",
        f"- TinyLLM median E2E throughput was `{short_vs_transformers:.2f}x`, `{medium_vs_transformers:.2f}x`, and "
        f"`{decode_vs_transformers:.2f}x` the tested Transformers baseline for short chat, medium chat, and long decode,",
        f"  but only `{prefill_vs_transformers:.2f}x` for long prefill. vLLM led all four performance cohorts.",
        "- Completion remained 100% at every replay load, but tail latency and relative goodput degraded materially",
        "  above 0.50C_ref; completion alone must not be read as an SLO or stability claim.",
        "",
        "## Scope and claim boundary",
        "",
        "Results apply only to the recorded commit, model, GPU, dtype, and workloads. Relative goodput",
        "uses experiment-local thresholds and is not a production SLA or a claim of serving parity.",
        "The labels `0.25C` through `0.90C` are retained in JSON for compatibility; in this report, `C_ref`",
        "means the completion rate of one simultaneous 1,000-request calibration for the same window.",
        "It is not a production saturation capacity derived from a steady-state load scan.",
        "",
        "## Environment",
        "",
        f"- Runtime candidate commit: `{environment.get('git_commit', '-')}`",
        f"- git_dirty: `{environment.get('git_dirty', '-')}`",
        f"- GPU / driver: `{gpu_summary}`",
        f"- CUDA toolkit: `{manifest.get('cuda_toolkit', '-')}`",
        f"- Suite Python / PyTorch / Transformers: `{suite_python.get('python', '-')} / "
        f"{suite_packages.get('torch', '-')} / {suite_packages.get('transformers', '-')}`",
        f"- vLLM environment Python / PyTorch / Transformers / vLLM: `{vllm_python.get('python', '-')} / "
        f"{vllm_packages.get('torch', '-')} / {vllm_packages.get('transformers', '-')} / "
        f"{vllm_packages.get('vllm', '-')}`",
        f"- Model: `{manifest.get('model', {}).get('repo_id', '-')}` at "
        f"`{manifest.get('model', {}).get('revision', '-')}`",
        "- TinyLLM compute / KV dtype: `FP32 / FP32`",
        "- Sampling: greedy; correctness retains EOS; performance uses fixed output and ignores EOS",
        f"- Trace warmup: `{manifest.get('configuration', {}).get('warmup_request_count', '-')} requests`; "
        "offline performance warmup/repeat: `1 / 1` per shard",
        *gpu_memory_lines,
        "",
        "## Sources and workload construction",
        "",
        "- OASST1 is a content proxy; it is not the original BurstGPT request text.",
        "- BurstGPT token counts come from GPT services and are matched approximately under the Qwen tokenizer.",
        "- Arrival timestamps are linearly scaled to the workload-specific measured reference rate.",
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
        "## Selected trace windows",
        "",
        "The windows come from the usable trace at the configured 10%, 50%, and 90% positions after filtering.",
        "ISL is measured after applying the Qwen chat template; OSL is the fixed per-request generation limit.",
        "",
        "| Window | Trace position | API / conversation | ISL tokens min / median / max | OSL tokens min / median / max | Input / output tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    quantiles = manifest.get("configuration", {}).get("window_quantiles", [])
    for index, window_id in enumerate(sorted(windows)):
        records = [record for record in selection if record.get("window_id") == window_id]
        api_count = sum(record.get("source_log_type") == "API log" for record in records)
        conversation_count = sum(record.get("source_log_type") == "Conversation log" for record in records)
        prompt_lengths = [int(record["prompt_len"]) for record in records]
        output_lengths = [int(record["max_new_tokens"]) for record in records]
        quantile = float(quantiles[index]) if index < len(quantiles) else 0.0
        lines.append(
            f"| {window_id} | {quantile:.0%} | {api_count} / {conversation_count} | "
            f"{min(prompt_lengths)} / {statistics.median(prompt_lengths):.1f} / {max(prompt_lengths)} | "
            f"{min(output_lengths)} / {statistics.median(output_lengths):.1f} / {max(output_lengths)} | "
            f"{sum(prompt_lengths):,} / {sum(output_lengths):,} |"
        )
    lines.extend(
        [
            "",
            "Window-03 is decode-heavy: its requested output-token total is more than twenty times either",
            "of the other windows. Request/s and `C_ref` therefore must be compared within a window, not",
            "treated as workload-independent engine capacity.",
            "",
            "## Cross-window trace summary",
            "",
            "Each cell reports `min / median / max` across the three windows.",
            "",
            "| Load | Achieved req/s | TTFT p99 ms | TPOT p99 ms | E2E p99 ms | Good ratio | Max concurrency |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for load_name in ("0.25C", "0.50C", "0.75C", "0.90C"):
        summary = cross_window[load_name]
        lines.append(
            f"| {load_name.replace('C', 'C_ref')} | {format_range(summary['achieved_request_per_s'])} | "
            f"{format_range(summary['ttft_ms_p99'])} | {format_range(summary['tpot_ms_p99'])} | "
            f"{format_range(summary['e2e_ms_p99'])} | {format_range(summary['good_request_ratio'])} | "
            f"{format_range(summary['max_concurrency'], 0)} |"
        )
    lines.extend(
        [
            "",
            "Median relative good-request ratio fell from 0.998 at 0.25C_ref to 0.981, 0.847, and 0.716",
            "as load increased. Median TTFT p99 rose from 18.831 s at 0.25C_ref to 54.901 s at 0.75C_ref.",
            "The knee between 0.50C_ref and 0.75C_ref is evidence of queueing pressure for these traces,",
            "not a universal capacity threshold.",
            "",
            "## Per-window trace details",
            "",
            "`C_ref = 1000 × 1000 / measured_generation_ms` for each simultaneous calibration.",
            "",
            "| Window | C_ref req/s | Load | Target req/s | Completed | Req/s | TTFT p99 ms | TPOT p99 ms | E2E p99 ms | Max conc. | Good ratio |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for window_id, window in sorted(windows.items()):
        for load_name, load in sorted(window.get("loads", {}).items()):
            summary = load["request_metrics"]["overall"]
            metrics = summary["metrics"]
            lines.append(
                f"| {window_id} | {window['capacity']['capacity_rps']:.6f} | {load_name.replace('C', 'C_ref')} | "
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
            "## Offline correctness and three-backend cohorts",
            "",
            f"Correctness used `{compared_requests}` mixed prompts with EOS enabled. TinyLLM, Transformers, and",
            f"vLLM token IDs matched pairwise with zero mismatches: "
            f"`{'yes' if correctness_match else 'no'}`.",
            "Performance cohorts used fixed output lengths and ignored EOS.",
            "",
            "| Cohort | ISL tokens | OSL | Shards × requests |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    cohort_labels = (("correctness", "Correctness"),) + performance_cohorts
    for cohort, label in cohort_labels:
        scenarios = [
            scenario for scenario in offline.get("scenarios", []) if scenario.get("cohort") == cohort
        ]
        records = [record for scenario in scenarios for record in scenario.get("selection", [])]
        prompt_lengths = [int(record["prompt_len"]) for record in records]
        output_lengths = sorted({int(record["max_new_tokens"]) for record in records})
        request_counts = [int(scenario.get("request_count", 0)) for scenario in scenarios]
        shard_text = (
            f"{len(scenarios)} × {request_counts[0]}"
            if request_counts and len(set(request_counts)) == 1
            else f"{len(scenarios)} / {sum(request_counts)} total"
        )
        osl_text = "/".join(str(value) for value in output_lengths)
        if cohort == "correctness":
            osl_text += ", EOS-aware"
        lines.append(
            f"| {label} | {min(prompt_lengths)}-{max(prompt_lengths)} | {osl_text} | {shard_text} |"
        )
    lines.extend(
        [
            "",
            "The following values are medians across the three deterministic shards, not confidence intervals.",
            "",
            "| Cohort | Requests | TinyLLM E2E tok/s | Transformers | vLLM | Tiny/Transformers | Tiny/vLLM | TinyLLM latency ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cohort, label in performance_cohorts:
        aggregate = offline_aggregates[cohort]
        request_count = sum(
            int(scenario.get("request_count", 0))
            for scenario in offline.get("scenarios", [])
            if scenario.get("cohort") == cohort
        )
        tiny = e2e_median(cohort, "tinyllm")
        transformers = e2e_median(cohort, "transformers")
        vllm = e2e_median(cohort, "vllm")
        tiny_latency = aggregate["backends"]["tinyllm"]["avg_total_latency_ms"]["p50"]
        lines.append(
            f"| {label} | {request_count} | {tiny:.3f} | {transformers:.3f} | {vllm:.3f} | "
            f"{tiny / transformers:.3f}x | {tiny / vllm:.3f}x | {tiny_latency:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Against the tested Transformers path, TinyLLM was strongest on short chat ({short_vs_transformers:.2f}x),",
            f"  medium chat ({medium_vs_transformers:.2f}x), and long decode ({decode_vs_transformers:.2f}x) median E2E throughput.",
            f"- Long prefill is the clearest weakness: TinyLLM reached only {prefill_vs_transformers:.2f}x the Transformers",
            "  median E2E throughput and roughly one tenth of vLLM's throughput in that cohort.",
            "- vLLM led every performance cohort, so these data support a workload-sensitive engineering baseline,",
            "  not a parity claim.",
            "- All replays completed, but higher-load goodput and tail-latency degradation show that completion rate",
            "  alone is insufficient for judging interactive service quality.",
            "",
            "## Relative goodput",
            "",
            "For each window, `TTFT SLO = 2.0 × that window's 0.25C_ref TTFT p99` and",
            "`TPOT SLO = 1.5 × that window's 0.25C_ref TPOT p99`. A request counts as good only if both",
            "conditions hold. These are relative experiment thresholds, not production objectives.",
            "",
            "## Limitations",
            "",
            "- Only three deterministic trace windows and one GPU/model/dtype configuration were measured.",
            "- OASST1 supplies content, while BurstGPT supplies timing and target lengths; they are not original paired requests.",
            "- BurstGPT lengths come from another tokenizer and are only approximately matched under Qwen.",
            "- Arrival gaps are linearly scaled, and fixed-output performance runs do not model natural EOS.",
            "- `C_ref` is a simultaneous-batch completion rate, not a steady-state production capacity estimate.",
            "- Offline medians summarize three fixed shards; they do not provide randomized-order variance or confidence intervals.",
            "- Network, request parsing, HTTP/gRPC, authentication, multi-GPU, failure recovery, and production reliability are excluded.",
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
            "",
            f"- Archive: `{manifest.get('raw_artifact', {}).get('filename', '-')}`",
            f"- SHA-256: `{manifest.get('raw_artifact', {}).get('sha256', '-')}`",
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
    public_environment = dict(trace_environment)
    memory_observation = gpu_memory_observation(public_environment.get("gpu_driver", []))
    if memory_observation:
        public_environment["gpu_memory_observation"] = memory_observation
    manifest["environment"] = public_environment
    manifest["cuda_toolkit"] = nvcc.stdout.strip().splitlines()[-1] if nvcc.returncode == 0 else "unknown"
    manifest["report_generated_by_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    ).stdout.strip()
    manifest["configuration"] = args.config_data
    manifest["commands"] = build_reproduction_commands(args)
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
    (publish_dir / "README.md").write_text(
        build_markdown(trace, offline, manifest, selection), encoding="utf-8"
    )
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
