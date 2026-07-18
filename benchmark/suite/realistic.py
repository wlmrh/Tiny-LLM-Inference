from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ISL_BUCKETS = ((1, 128), (129, 512), (513, 2048), (2049, 3584))
OSL_BUCKETS = ((1, 32), (33, 128), (129, 512))
WINDOW_QUANTILES = (0.10, 0.50, 0.90)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_ids(tokenizer: Any, text: str) -> List[int]:
    return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=True)]


def _tree_root(record: Dict[str, Any]) -> Dict[str, Any]:
    root = record.get("prompt")
    return root if isinstance(root, dict) else record


def load_oasst_candidates(path: Path, tokenizer: Any, max_prompt_tokens: int = 4095) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            tree = json.loads(line)
            root = _tree_root(tree)
            tree_id = str(root.get("message_tree_id") or root.get("message_id") or f"tree-{line_number}")

            def visit(node: Dict[str, Any], messages: List[Dict[str, str]]) -> None:
                source_role = str(node.get("role", ""))
                role = "user" if source_role == "prompter" else "assistant" if source_role == "assistant" else ""
                text = str(node.get("text", "")).strip()
                if not role or not text or bool(node.get("deleted", False)) or node.get("review_result") is False:
                    return
                current = [*messages, {"role": role, "content": text}]
                if role == "user":
                    prompt = tokenizer.apply_chat_template(current, tokenize=False, add_generation_prompt=True)
                    ids = _token_ids(tokenizer, prompt)
                    if 0 < len(ids) <= max_prompt_tokens:
                        candidates.append(
                            {
                                "prompt": prompt,
                                "input_ids": ids,
                                "prompt_len": len(ids),
                                "tree_id": tree_id,
                                "message_id": str(node.get("message_id", "")),
                                "language": str(node.get("lang", "unknown")),
                            }
                        )
                for reply in node.get("replies") or []:
                    if isinstance(reply, dict):
                        visit(reply, current)

            visit(root, [])
    candidates.sort(key=lambda item: (int(item["prompt_len"]), str(item["message_id"])))
    if not candidates:
        raise RuntimeError(f"no usable OASST candidates found in {path}")
    return candidates


def _column(row: Dict[str, str], *names: str) -> str:
    normalized = {key.strip().lower().replace("_", " "): value for key, value in row.items() if key is not None}
    for name in names:
        key = name.strip().lower().replace("_", " ")
        if key in normalized:
            return str(normalized[key]).strip()
    raise RuntimeError(f"missing BurstGPT column; expected one of {names}")


def load_burstgpt_rows(path: Path, max_output_tokens: int = 512, max_total_tokens: int = 4096) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for source_index, row in enumerate(reader):
            request_tokens = int(float(_column(row, "Request tokens", "Request_tokens")))
            response_tokens = int(float(_column(row, "Response tokens", "Response_tokens")))
            if request_tokens < 1 or response_tokens < 1 or response_tokens > max_output_tokens:
                continue
            if request_tokens + response_tokens > max_total_tokens:
                continue
            rows.append(
                {
                    "source_trace_index": source_index,
                    "source_timestamp_s": float(_column(row, "Timestamp")),
                    "source_request_tokens": request_tokens,
                    "source_response_tokens": response_tokens,
                    "source_log_type": _column(row, "Log Type", "Log_Type"),
                    "source_model": _column(row, "Model"),
                    "source_session_sha256": sha256_text(_column(row, "Session ID", "Session_ID")),
                }
            )
    rows.sort(key=lambda item: (float(item["source_timestamp_s"]), int(item["source_trace_index"])))
    if not rows:
        raise RuntimeError(f"no usable BurstGPT rows found in {path}")
    return rows


class CandidateMatcher:
    def __init__(self, candidates: Sequence[Dict[str, Any]]) -> None:
        self._by_length: Dict[int, List[Dict[str, Any]]] = {}
        for candidate in candidates:
            self._by_length.setdefault(int(candidate["prompt_len"]), []).append(candidate)
        for values in self._by_length.values():
            values.sort(key=lambda item: str(item["message_id"]))
        self._used: set[str] = set()

    def match(self, target_tokens: int, max_prompt_tokens: int) -> Dict[str, Any] | None:
        tolerance = max(16, int(math.ceil(target_tokens * 0.10)))
        lower = max(1, target_tokens - tolerance)
        upper = min(max_prompt_tokens, target_tokens + tolerance)
        best: tuple[int, str, Dict[str, Any]] | None = None
        for length in range(lower, upper + 1):
            for candidate in self._by_length.get(length, []):
                message_id = str(candidate["message_id"])
                if message_id in self._used:
                    continue
                choice = (abs(length - target_tokens), message_id, candidate)
                if best is None or choice[:2] < best[:2]:
                    best = choice
                break
        if best is None:
            return None
        self._used.add(best[1])
        return best[2]


def select_trace_windows(
    rows: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    window_size: int = 1000,
    quantiles: Sequence[float] = WINDOW_QUANTILES,
    max_total_tokens: int = 4096,
) -> List[List[Dict[str, Any]]]:
    if len(rows) < window_size:
        raise RuntimeError(f"trace has only {len(rows)} usable rows; need at least {window_size}")
    matcher = CandidateMatcher(candidates)
    used_trace_indices: set[int] = set()
    windows: List[List[Dict[str, Any]]] = []
    for window_number, quantile in enumerate(quantiles, start=1):
        start = int(math.floor((len(rows) - window_size) * float(quantile)))
        selected: List[Dict[str, Any]] = []
        cursor = start
        while cursor < len(rows) and len(selected) < window_size:
            row = rows[cursor]
            cursor += 1
            trace_index = int(row["source_trace_index"])
            if trace_index in used_trace_indices:
                continue
            max_prompt = max_total_tokens - int(row["source_response_tokens"])
            candidate = matcher.match(int(row["source_request_tokens"]), max_prompt)
            if candidate is None:
                continue
            request_id = f"window-{window_number:02d}-{len(selected):04d}"
            record = {
                "request_id": request_id,
                "prompt": candidate["prompt"],
                "input_ids": candidate["input_ids"],
                "prompt_len": int(candidate["prompt_len"]),
                "max_new_tokens": int(row["source_response_tokens"]),
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "ignore_eos": True,
                "arrival_ms": 0.0,
                **row,
                "oasst_tree_id": candidate["tree_id"],
                "oasst_message_id": candidate["message_id"],
                "language": candidate["language"],
                "prompt_sha256": sha256_text(str(candidate["prompt"])),
                "window_id": f"window-{window_number:02d}",
            }
            selected.append(record)
            used_trace_indices.add(trace_index)
        if len(selected) != window_size:
            raise RuntimeError(
                f"window {window_number} produced {len(selected)} matched requests; expected {window_size}"
            )
        first_timestamp = float(selected[0]["source_timestamp_s"])
        for record in selected:
            record["arrival_ms"] = (float(record["source_timestamp_s"]) - first_timestamp) * 1000.0
        windows.append(selected)
    return windows


def scale_trace_arrivals(records: Sequence[Dict[str, Any]], target_rate_rps: float) -> List[Dict[str, Any]]:
    if target_rate_rps <= 0.0 or len(records) < 2:
        raise ValueError("target_rate_rps must be positive and at least two records are required")
    first = float(records[0]["arrival_ms"])
    last = float(records[-1]["arrival_ms"])
    source_duration_ms = last - first
    if source_duration_ms <= 0.0:
        raise ValueError("source trace duration must be positive")
    target_duration_ms = (len(records) - 1) * 1000.0 / target_rate_rps
    factor = target_duration_ms / source_duration_ms
    scaled: List[Dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["arrival_ms"] = (float(record["arrival_ms"]) - first) * factor
        scaled.append(item)
    return scaled


def write_workload_jsonl(path: Path, records: Iterable[Dict[str, Any]], include_prompts: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            item = dict(record)
            if not include_prompts:
                item.pop("prompt", None)
                item.pop("input_ids", None)
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_workload_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bucket_name(value: int, buckets: Sequence[tuple[int, int]]) -> str:
    for lower, upper in buckets:
        if lower <= value <= upper:
            return f"{lower}-{upper}"
    return "out-of-range"


def public_selection(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    omitted = {"prompt", "input_ids"}
    return [{key: value for key, value in record.items() if key not in omitted} for record in records]


def _evenly_spaced(items: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    if len(items) < count:
        raise RuntimeError(f"cohort has only {len(items)} eligible prompts; need {count}")
    if count == 1:
        return [dict(items[len(items) // 2])]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [dict(items[index]) for index in indices]


def build_offline_cohorts(records: Sequence[Dict[str, Any]]) -> Dict[str, List[List[Dict[str, Any]]]]:
    definitions = {
        "correctness": (1, 3584, 32, 8, 1, False),
        "short_chat": (32, 256, 64, 8, 3, True),
        "medium_chat": (257, 1024, 128, 8, 3, True),
        "long_prefill": (1025, 3584, 64, 4, 3, True),
        "long_decode": (32, 1024, 256, 8, 3, True),
    }
    cohorts: Dict[str, List[List[Dict[str, Any]]]] = {}
    for name, (minimum, maximum, output_tokens, batch_size, shard_count, ignore_eos) in definitions.items():
        eligible = sorted(
            (item for item in records if minimum <= int(item["prompt_len"]) <= maximum),
            key=lambda item: (int(item["prompt_len"]), str(item["request_id"])),
        )
        selected = _evenly_spaced(eligible, batch_size * shard_count)
        shards: List[List[Dict[str, Any]]] = []
        for shard_index in range(shard_count):
            shard = []
            for item_index, original in enumerate(
                selected[shard_index * batch_size : (shard_index + 1) * batch_size]
            ):
                item = dict(original)
                item["request_id"] = f"{name}-{shard_index + 1:02d}-{item_index:03d}"
                item["max_new_tokens"] = output_tokens
                item["ignore_eos"] = ignore_eos
                item["arrival_ms"] = 0.0
                shard.append(item)
            shards.append(shard)
        cohorts[name] = shards
    return cohorts
