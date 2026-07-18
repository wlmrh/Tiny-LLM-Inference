#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from suite.realistic import (
    load_burstgpt_rows,
    load_oasst_candidates,
    public_selection,
    select_trace_windows,
    sha256_file,
    write_workload_jsonl,
)
from suite.workloads import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic realistic TinyLLM workloads")
    parser.add_argument("--config", default="benchmark/configs/qwen25_realistic_v1.json")
    parser.add_argument("--burstgpt", required=True)
    parser.add_argument("--oasst", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-size", type=int, default=1000)
    return parser.parse_args()


def file_record(path: Path) -> dict:
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> int:
    args = parse_args()
    burst_path = Path(args.burstgpt)
    oasst_path = Path(args.oasst)
    output_dir = Path(args.output_dir)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    sources = config["sources"]
    burst_record = file_record(burst_path)
    oasst_record = file_record(oasst_path)
    for name, actual in (("burstgpt", burst_record), ("oasst1", oasst_record)):
        expected = str(sources[name]["sha256"])
        if actual["sha256"] != expected:
            raise RuntimeError(
                f"{name} SHA-256 mismatch: expected {expected}, got {actual['sha256']}"
            )
    tokenizer = load_tokenizer(Path(args.model_dir))
    candidates = load_oasst_candidates(oasst_path, tokenizer)
    trace_rows = load_burstgpt_rows(burst_path)
    windows = select_trace_windows(trace_rows, candidates, window_size=args.window_size)

    selection = []
    workload_files = []
    for index, records in enumerate(windows, start=1):
        path = output_dir / "base" / f"window-{index:02d}.jsonl"
        write_workload_jsonl(path, records)
        workload_files.append(file_record(path))
        selection.extend(public_selection(records))
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selection.json"
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    model_dir = Path(args.model_dir)
    model_files = [model_dir / "config.json", model_dir / "tokenizer.json"]
    model_files.extend(
        path
        for path in (model_dir / "tokenizer_config.json", model_dir / "generation_config.json")
        if path.is_file()
    )
    model_files.extend(sorted(model_dir.glob("*.safetensors")))
    missing_model_files = [str(path) for path in model_files if not path.is_file()]
    if missing_model_files:
        raise RuntimeError(f"missing model files: {', '.join(missing_model_files)}")
    manifest = {
        "schema_version": 1,
        "benchmark": "tinyllm_realistic_v1",
        "sources": sources,
        "burstgpt": {**sources["burstgpt"], **burst_record},
        "oasst1": {**sources["oasst1"], **oasst_record},
        "model": {
            **sources["model"],
            "local_path": str(model_dir),
            "files": [file_record(path) for path in model_files],
        },
        "oasst_candidates": len(candidates),
        "usable_trace_rows": len(trace_rows),
        "window_size": args.window_size,
        "window_count": len(windows),
        "selection": file_record(selection_path),
        "private_workload_files": workload_files,
    }
    (output_dir / "workload-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir.parent / "dataset-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
