#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_PROMPTS = ["hello", "tiny llm inference"]


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TinyLLM and Transformers generation benchmark results")
    parser.add_argument("--tinyllm-binary", default="build/benchmark/llama_engine_benchmark")
    parser.add_argument("--backend", choices=("tinyllm", "transformers", "all"), default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=non_negative_int, default=1)
    parser.add_argument("--repeat", type=positive_int, default=3)
    parser.add_argument("--max-new-tokens", type=positive_int, default=8)
    parser.add_argument("--prompt", action="append", dest="prompts", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("model_dir")
    return parser.parse_args()


def command_common_args(args: argparse.Namespace) -> List[str]:
    command = [
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    for prompt in args.prompts or DEFAULT_PROMPTS:
        command.extend(["--prompt", prompt])
    command.append("--json")
    command.append(str(Path(args.model_dir).expanduser()))
    return command


def parse_json_line(stdout: str, backend: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        data = json.loads(line)
        if "backend" not in data:
            data["backend"] = backend
        return data
    raise RuntimeError(f"{backend} benchmark did not emit a JSON object")


def run_command(command: List[str], backend: str) -> Dict[str, Any]:
    run = subprocess.run(command, check=False, capture_output=True, text=True)
    if run.returncode != 0:
        sys.stderr.write(run.stdout)
        sys.stderr.write(run.stderr)
        raise RuntimeError(f"{backend} benchmark failed with exit code {run.returncode}: {' '.join(command)}")
    return parse_json_line(run.stdout, backend)


def run_tinyllm(args: argparse.Namespace) -> Dict[str, Any]:
    binary = Path(args.tinyllm_binary)
    if not binary.exists():
        raise RuntimeError(f"TinyLLM benchmark binary does not exist: {binary}; build it first with cmake --build")
    return run_command([str(binary), *command_common_args(args)], "tinyllm")


def run_transformers(args: argparse.Namespace) -> Dict[str, Any]:
    script = Path(__file__).resolve().with_name("transformers_generate_benchmark.py")
    return run_command([sys.executable, str(script), *command_common_args(args)], "transformers")


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def build_comparison(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_backend = {str(item["backend"]): item for item in results}
    comparison: Dict[str, Any] = {
        "benchmark": "benchmark_comparison",
        "results": results,
        "ratios": {},
    }
    tiny = by_backend.get("tinyllm")
    hf = by_backend.get("transformers")
    if tiny is not None and hf is not None:
        comparison["ratios"] = {
            "tinyllm_vs_transformers_latency": ratio(float(tiny["avg_total_latency_ms"]), float(hf["avg_total_latency_ms"])),
            "tinyllm_vs_transformers_first_token_latency": ratio(
                float(tiny["avg_first_token_latency_ms"]), float(hf["avg_first_token_latency_ms"])
            ),
            "tinyllm_vs_transformers_e2e_throughput": ratio(
                float(tiny["end_to_end_tokens_per_s"]), float(hf["end_to_end_tokens_per_s"])
            ),
            "tinyllm_vs_transformers_decode_throughput": ratio(
                float(tiny["decode_tokens_per_s"]), float(hf["decode_tokens_per_s"])
            ),
            "tinyllm_vs_transformers_load_init": ratio(float(tiny["avg_load_init_ms"]), float(hf["avg_load_init_ms"])),
        }
    return comparison


def print_table(results: List[Dict[str, Any]], comparison: Dict[str, Any]) -> None:
    print("benchmark_comparison")
    print("backend       load_ms    latency_ms    first_ms    e2e_tok_s    decode_tok_s    gen_tokens")
    for item in results:
        print(
            f"{item['backend']:<12} "
            f"{float(item['avg_load_init_ms']):>8.3f} "
            f"{float(item['avg_total_latency_ms']):>13.3f} "
            f"{float(item['avg_first_token_latency_ms']):>10.3f} "
            f"{float(item['end_to_end_tokens_per_s']):>11.3f} "
            f"{float(item['decode_tokens_per_s']):>14.3f} "
            f"{float(item['avg_generated_tokens']):>10.3f}"
        )
    ratios = comparison.get("ratios", {})
    if ratios:
        print("ratios tinyllm/transformers")
        print(f"  latency: {ratios['tinyllm_vs_transformers_latency']:.3f}")
        print(f"  first_token_latency: {ratios['tinyllm_vs_transformers_first_token_latency']:.3f}")
        print(f"  e2e_throughput: {ratios['tinyllm_vs_transformers_e2e_throughput']:.3f}")
        print(f"  decode_throughput: {ratios['tinyllm_vs_transformers_decode_throughput']:.3f}")
        print(f"  load_init: {ratios['tinyllm_vs_transformers_load_init']:.3f}")


def main() -> int:
    args = parse_args()
    results: List[Dict[str, Any]] = []
    if args.backend in ("tinyllm", "all"):
        results.append(run_tinyllm(args))
    if args.backend in ("transformers", "all"):
        results.append(run_transformers(args))
    comparison = build_comparison(results)
    print_table(results, comparison)
    if args.json:
        print(json.dumps(comparison, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"run_benchmark_comparison failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
