from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROMPT_FRAGMENTS = [
    "The benchmark should reflect production-like serving pressure with deterministic greedy decoding. ",
    "Discuss scheduler fairness, paged KV cache reuse, chunked prefill, and decode throughput. ",
    "Use concrete AI infrastructure tradeoffs and include both English and Chinese text. ",
    "请分析连续批处理、长上下文预填充、KV 缓存、解码吞吐和显存压力。 ",
]


def load_tokenizer(model_dir: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(f"missing transformers dependency for workload generation: {exc}") from exc
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)


def token_ids(tokenizer: Any, text: str) -> List[int]:
    return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=True)]


def make_prompt(tokenizer: Any, target_tokens: int, seed: int) -> Tuple[str, List[int]]:
    fragments = [PROMPT_FRAGMENTS[(seed + idx) % len(PROMPT_FRAGMENTS)] for idx in range(len(PROMPT_FRAGMENTS))]
    text = f"Request {seed}: "
    idx = 0
    while len(token_ids(tokenizer, text)) < target_tokens:
        text += fragments[idx % len(fragments)]
        idx += 1

    lo = 1
    hi = len(text)
    best = text
    best_ids = token_ids(tokenizer, best)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        ids = token_ids(tokenizer, candidate)
        if len(ids) <= target_tokens:
            best = candidate
            best_ids = ids
            lo = mid + 1
        else:
            hi = mid - 1

    return best, best_ids


def write_workload_jsonl(
    output_path: Path,
    tokenizer: Any,
    scenario: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch = int(scenario["batch"])
    input_tokens = int(scenario["input_tokens"])
    max_new_tokens = int(scenario["output_tokens"])
    records: List[Dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index in range(batch):
            prompt, ids = make_prompt(tokenizer, input_tokens, index)
            record = {
                "request_id": f"{scenario['name']}-{index:04d}",
                "prompt": prompt,
                "input_ids": ids,
                "prompt_len": len(ids),
                "max_new_tokens": max_new_tokens,
                "temperature": float(scenario.get("temperature", defaults.get("temperature", 0.0))),
                "top_p": float(scenario.get("top_p", defaults.get("top_p", 1.0))),
                "top_k": int(scenario.get("top_k", defaults.get("top_k", 0))),
                "repetition_penalty": float(
                    scenario.get("repetition_penalty", defaults.get("repetition_penalty", 1.0))
                ),
                "ignore_eos": bool(scenario.get("ignore_eos", defaults.get("ignore_eos", False))),
                "arrival_ms": float(scenario.get("arrival_ms", defaults.get("arrival_ms", 0.0))),
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    return {
        "path": str(output_path),
        "request_count": len(records),
        "target_input_tokens": input_tokens,
        "actual_input_tokens": sum(len(item["input_ids"]) for item in records),
        "max_new_tokens": max_new_tokens,
    }

