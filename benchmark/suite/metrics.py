from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from suite.realistic import ISL_BUCKETS, OSL_BUCKETS, bucket_name


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_values(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "avg": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def read_event_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"available": False, "requests": [], "summary": {}}

    grouped: Dict[Tuple[int, str], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            key = (int(event.get("repeat", 0)), str(event["request_id"]))
            item = grouped.setdefault(
                key,
                {
                    "repeat": key[0],
                    "request_id": key[1],
                    "prompt_index": int(event.get("prompt_index", 0)),
                    "submit_ms": 0.0,
                    "admit_ms": -1.0,
                    "first_token_ms": -1.0,
                    "finish_ms": -1.0,
                    "generated_tokens": 0,
                    "requested_tokens": int(event.get("requested_tokens", 0)),
                    "finish_reason": "",
                    "_event_counts": {"submit": 0, "admit": 0, "token": 0, "finish": 0},
                },
            )
            event_type = str(event.get("event", ""))
            if event_type in item["_event_counts"]:
                item["_event_counts"][event_type] += 1
            if event_type == "submit":
                item["submit_ms"] = float(event.get("time_ms", 0.0))
                item["requested_tokens"] = int(event.get("requested_tokens", item["requested_tokens"]))
            elif event_type == "admit":
                item["admit_ms"] = float(event.get("time_ms", 0.0))
            elif event_type == "token":
                item["generated_tokens"] += 1
                if item["first_token_ms"] < 0.0:
                    item["first_token_ms"] = float(event.get("time_ms", 0.0))
            elif event_type == "finish":
                item["finish_ms"] = float(event.get("time_ms", 0.0))
                item["finish_reason"] = str(event.get("finish_reason", ""))
                item["generated_tokens"] = int(event.get("generated_tokens", item["generated_tokens"]))

    requests = []
    ttft_ms: List[float] = []
    engine_ttft_ms: List[float] = []
    queue_ms: List[float] = []
    e2e_ms: List[float] = []
    tpot_ms: List[float] = []
    completeness_errors: List[str] = []
    for item in sorted(grouped.values(), key=lambda value: (value["repeat"], value["prompt_index"])):
        event_counts = item.pop("_event_counts")
        for event_type in ("submit", "admit", "finish"):
            if event_counts[event_type] != 1:
                completeness_errors.append(
                    f"repeat={item['repeat']} request_id={item['request_id']} expected one {event_type} event, "
                    f"found {event_counts[event_type]}"
                )
        first_token_ms = float(item["first_token_ms"])
        finish_ms = float(item["finish_ms"])
        submit_ms = float(item["submit_ms"])
        admit_ms = float(item["admit_ms"])
        if admit_ms < 0.0:
            admit_ms = submit_ms
            item["admit_ms"] = admit_ms
        generated_tokens = int(item["generated_tokens"])
        requested_tokens = int(item["requested_tokens"])
        if event_counts["token"] != generated_tokens:
            completeness_errors.append(
                f"repeat={item['repeat']} request_id={item['request_id']} token event count "
                f"{event_counts['token']} != generated_tokens {generated_tokens}"
            )
        if requested_tokens > 0 and generated_tokens != requested_tokens:
            completeness_errors.append(
                f"repeat={item['repeat']} request_id={item['request_id']} generated_tokens "
                f"{generated_tokens} != requested_tokens {requested_tokens}"
            )
        request_ttft = max(0.0, first_token_ms - submit_ms) if first_token_ms >= 0.0 else 0.0
        request_engine_ttft = max(0.0, first_token_ms - admit_ms) if first_token_ms >= 0.0 else 0.0
        request_queue = max(0.0, admit_ms - submit_ms)
        request_e2e = max(0.0, finish_ms - submit_ms) if finish_ms >= 0.0 else 0.0
        request_tpot = (
            max(0.0, finish_ms - first_token_ms) / float(generated_tokens - 1)
            if generated_tokens > 1 and first_token_ms >= 0.0 and finish_ms >= first_token_ms
            else 0.0
        )
        item["ttft_ms"] = request_ttft
        item["engine_ttft_ms"] = request_engine_ttft
        item["queue_ms"] = request_queue
        item["e2e_ms"] = request_e2e
        item["tpot_ms"] = request_tpot
        requests.append(item)
        ttft_ms.append(request_ttft)
        engine_ttft_ms.append(request_engine_ttft)
        queue_ms.append(request_queue)
        e2e_ms.append(request_e2e)
        tpot_ms.append(request_tpot)

    return {
        "available": True,
        "complete": bool(grouped) and not completeness_errors,
        "completeness_errors": completeness_errors,
        "path": str(path),
        "requests": requests,
        "summary": {
            "ttft_ms": summarize_values(ttft_ms),
            "engine_ttft_ms": summarize_values(engine_ttft_ms),
            "queue_ms": summarize_values(queue_ms),
            "e2e_ms": summarize_values(e2e_ms),
            "tpot_ms": summarize_values(tpot_ms),
        },
    }


def summarize_requests(requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = ("queue_ms", "ttft_ms", "engine_ttft_ms", "e2e_ms", "tpot_ms")
    generated = sum(int(item.get("generated_tokens", 0)) for item in requests)
    requested = sum(int(item.get("requested_tokens", item.get("max_new_tokens", 0))) for item in requests)
    prompt_tokens = sum(int(item.get("prompt_len", item.get("prompt_tokens", 0))) for item in requests)
    submit_times = [float(item.get("submit_ms", 0.0)) for item in requests]
    finish_times = [float(item.get("finish_ms", -1.0)) for item in requests]
    valid_finishes = [value for value in finish_times if value >= 0.0]
    start_ms = min(submit_times, default=0.0)
    end_ms = max(valid_finishes, default=start_ms)
    duration_ms = max(0.0, end_ms - start_ms)
    completed = sum(1 for item in requests if float(item.get("finish_ms", -1.0)) >= 0.0)
    timeline = []
    for item in requests:
        submit_ms = float(item.get("submit_ms", 0.0))
        finish_ms = float(item.get("finish_ms", -1.0))
        timeline.append((submit_ms, 1))
        if finish_ms >= submit_ms:
            timeline.append((finish_ms, -1))
    active = 0
    max_concurrency = 0
    for _, delta in sorted(timeline, key=lambda event: (event[0], event[1])):
        active = max(0, active + delta)
        max_concurrency = max(max_concurrency, active)
    return {
        "request_count": len(requests),
        "completed_request_count": completed,
        "error_count": len(requests) - completed,
        "metrics": {
            name: summarize_values([float(item.get(name, 0.0)) for item in requests]) for name in metric_names
        },
        "request_per_s": len(requests) * 1000.0 / duration_ms if duration_ms > 0.0 else 0.0,
        "input_tokens_per_s": prompt_tokens * 1000.0 / duration_ms if duration_ms > 0.0 else 0.0,
        "output_tokens_per_s": generated * 1000.0 / duration_ms if duration_ms > 0.0 else 0.0,
        "total_tokens_per_s": (prompt_tokens + generated) * 1000.0 / duration_ms if duration_ms > 0.0 else 0.0,
        "requested_output_tokens": requested,
        "generated_output_tokens": generated,
        "wall_clock_ms": duration_ms,
        "max_concurrency": max_concurrency,
    }


def enrich_and_group_requests(
    requests: List[Dict[str, Any]], workload_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    metadata = {str(item["request_id"]): item for item in workload_records}
    enriched: List[Dict[str, Any]] = []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for request in requests:
        item = dict(request)
        source = metadata.get(str(item["request_id"]), {})
        for key in (
            "prompt_len",
            "max_new_tokens",
            "source_log_type",
            "language",
            "window_id",
            "prompt_sha256",
        ):
            if key in source:
                item[key] = source[key]
        enriched.append(item)
        labels = (
            f"log_type:{item.get('source_log_type', 'unknown')}",
            f"isl:{bucket_name(int(item.get('prompt_len', 0)), ISL_BUCKETS)}",
            f"osl:{bucket_name(int(item.get('max_new_tokens', 0)), OSL_BUCKETS)}",
        )
        for label in labels:
            groups.setdefault(label, []).append(item)
    return {
        "requests": enriched,
        "overall": summarize_requests(enriched),
        "groups": {name: summarize_requests(values) for name, values in sorted(groups.items())},
    }


def relative_goodput(
    requests: List[Dict[str, Any]], ttft_slo_ms: float, tpot_slo_ms: float
) -> Dict[str, float]:
    good = sum(
        1
        for item in requests
        if float(item.get("ttft_ms", 0.0)) <= ttft_slo_ms
        and float(item.get("tpot_ms", 0.0)) <= tpot_slo_ms
    )
    duration_ms = max((float(item.get("finish_ms", 0.0)) for item in requests), default=0.0)
    return {
        "good_requests": float(good),
        "good_request_ratio": good / len(requests) if requests else 0.0,
        "goodput_request_per_s": good * 1000.0 / duration_ms if duration_ms > 0.0 else 0.0,
        "ttft_slo_ms": ttft_slo_ms,
        "tpot_slo_ms": tpot_slo_ms,
    }
