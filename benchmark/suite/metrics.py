from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


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
                    "finish_reason": "",
                    "_event_counts": {"submit": 0, "admit": 0, "token": 0, "finish": 0},
                },
            )
            event_type = str(event.get("event", ""))
            if event_type in item["_event_counts"]:
                item["_event_counts"][event_type] += 1
            if event_type == "submit":
                item["submit_ms"] = float(event.get("time_ms", 0.0))
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
        if event_counts["token"] != generated_tokens:
            completeness_errors.append(
                f"repeat={item['repeat']} request_id={item['request_id']} token event count "
                f"{event_counts['token']} != generated_tokens {generated_tokens}"
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
