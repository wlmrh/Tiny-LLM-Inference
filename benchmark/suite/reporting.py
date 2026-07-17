from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _backend_rows(results: List[Dict[str, Any]]) -> List[str]:
    rows = [
        "| Backend | Load ms | Latency ms | TTFT ms | E2E tok/s | Decode tok/s | Gen tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        rows.append(
            "| {backend} | {load} | {latency} | {ttft} | {e2e} | {decode} | {tokens} |".format(
                backend=item.get("backend", "-"),
                load=_fmt(item.get("avg_load_init_ms")),
                latency=_fmt(item.get("avg_total_latency_ms")),
                ttft=_fmt(item.get("avg_first_token_latency_ms")),
                e2e=_fmt(item.get("end_to_end_tokens_per_s")),
                decode=_fmt(item.get("decode_tokens_per_s")),
                tokens=_fmt(item.get("avg_generated_tokens")),
            )
        )
    return rows


def _event_rows(summary: Dict[str, Any]) -> List[str]:
    if not summary.get("available"):
        return ["TinyLLM request event trace was not available."]
    stats = summary.get("summary", {})
    rows = []
    if not summary.get("complete", False):
        rows.append("Event trace incomplete: `" + "; ".join(summary.get("completeness_errors", [])) + "`")
        rows.append("")
    rows.extend([
        "| Metric | Avg | P50 | P95 | P99 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for name in ("queue_ms", "ttft_ms", "engine_ttft_ms", "e2e_ms", "tpot_ms"):
        item = stats.get(name, {})
        rows.append(
            f"| {name} | {_fmt(item.get('avg'))} | {_fmt(item.get('p50'))} | "
            f"{_fmt(item.get('p95'))} | {_fmt(item.get('p99'))} |"
        )
    return rows


def _environment_rows(environment: Dict[str, Any]) -> List[str]:
    rows = [
        f"- git_commit: `{environment.get('git_commit', '-')}`",
        f"- git_dirty: `{environment.get('git_dirty', '-')}`",
        f"- GPU/driver: `{' | '.join(environment.get('gpu_driver', [])) or '-'}`",
    ]
    for name, snapshot in environment.get("python", {}).items():
        rows.append(
            f"- Python ({name}): `{snapshot.get('executable', '-')}` / `{snapshot.get('python', '-')}` / "
            f"packages `{json.dumps(snapshot.get('packages', {}), sort_keys=True)}`"
        )
    return rows


def build_markdown(report: Dict[str, Any]) -> str:
    lines = [
        f"# TinyLLM Benchmark Suite: {report.get('label', '-')}",
        "",
        f"- generated_at: `{report.get('generated_at', '-')}`",
        f"- model_dir: `{report.get('model_dir', '-')}`",
        f"- device: `{report.get('device', '-')}`",
        f"- config: `{report.get('config_path', '-')}`",
        f"- invocation: `{json.dumps(report.get('invocation', []), ensure_ascii=False)}`",
        "",
        "## Environment",
        "",
        *_environment_rows(report.get("environment", {})),
        "",
    ]
    for scenario in report.get("scenarios", []):
        workload = scenario.get("workload", {})
        lines.extend(
            [
                f"## {workload.get('name', '-')}",
                "",
                f"- mode: `{scenario.get('benchmark_mode', '-')}`",
                f"- traffic_mode: `{scenario.get('traffic_mode', '-')}`",
                f"- batch: `{workload.get('batch', '-')}`",
                f"- target ISL/OSL: `{workload.get('input_tokens', '-')}/{workload.get('output_tokens', '-')}`",
                f"- workload_jsonl: `{scenario.get('workload_jsonl', '-')}`",
                "",
            ]
        )
        if scenario.get("error"):
            lines.extend([f"Run failed: `{scenario['error']}`", ""])
            continue
        lines.extend(_backend_rows(scenario.get("results", [])))
        lines.append("")
        ratios = scenario.get("ratios", {})
        if ratios:
            lines.append("Ratios:")
            for key, value in sorted(ratios.items()):
                lines.append(f"- `{key}`: `{_fmt(value)}`")
            lines.append("")
        skipped = scenario.get("skipped_backends", [])
        if skipped:
            lines.append(f"Skipped backends: `{', '.join(skipped)}`")
            lines.append("")
        agreement = scenario.get("output_agreement", {})
        if agreement:
            lines.append("Output token agreement:")
            for key, value in sorted(agreement.items()):
                lines.append(
                    f"- `{key}`: match=`{value.get('match', False)}`, "
                    f"mismatches=`{value.get('mismatch_count', 0)}`"
                )
            lines.append("")
        lines.extend(_event_rows(scenario.get("tinyllm_events", {})))
        lines.append("")
    return "\n".join(lines)


def write_reports(output_dir: Path, label: str, report: Dict[str, Any]) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{label}.json"
    md_path = output_dir / f"{label}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(report) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
