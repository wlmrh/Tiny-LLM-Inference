#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_PROMPTS = ["hello", "tiny llm inference"]
VALID_BACKENDS = ("tinyllm", "transformers", "vllm")


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
    parser = argparse.ArgumentParser(description="Compare TinyLLM, Transformers, and vLLM generation benchmark results")
    parser.add_argument("--tinyllm-binary", default="build/benchmark/llama_engine_benchmark")
    parser.add_argument(
        "--backend",
        default="all",
        help="backend to run: all, tinyllm, transformers, vllm, or a comma-separated subset",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tinyllm-dtype", default="fp32", choices=("fp32", "bf16"))
    parser.add_argument("--tinyllm-kv-cache-dtype", default="fp32", choices=("fp32", "bf16"))
    parser.add_argument("--warmup", type=non_negative_int, default=1)
    parser.add_argument("--repeat", type=positive_int, default=3)
    parser.add_argument("--max-new-tokens", type=positive_int, default=8)
    parser.add_argument("--ignore-eos", action="store_true", help="require backends to generate max_new_tokens")
    parser.add_argument("--workload-jsonl", help="flat JSONL workload with prompt/request_id records")
    parser.add_argument("--events-jsonl", help="TinyLLM raw request event JSONL output path")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=non_negative_int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--benchmark-mode",
        choices=("correctness", "fixed_output_perf"),
        default="correctness",
    )
    parser.add_argument("--max-num-batched-token-cap", type=positive_int, default=4096)
    parser.add_argument("--vllm-python", default=sys.executable, help="Python executable used for the vLLM baseline")
    parser.add_argument("--vllm-dtype", default="auto", help="vLLM dtype, for example auto, float16, bfloat16, or float32")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--vllm-max-model-len",
        type=non_negative_int,
        default=4096,
        help="vLLM max_model_len; use 0 to leave the model config default unchanged",
    )
    parser.add_argument("--vllm-enforce-eager", action="store_true", help="disable vLLM CUDA graph capture")
    parser.add_argument("--prompt", action="append", dest="prompts", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--profile-detail", action="store_true", help="enable TinyLLM detailed runtime profiling")
    parser.add_argument("model_dir")
    args = parser.parse_args()
    args.selected_backends = parse_backend_selection(args.backend)
    return args


def parse_backend_selection(text: str) -> List[str]:
    if text == "all":
        return list(VALID_BACKENDS)
    selected: List[str] = []
    for item in text.split(","):
        backend = item.strip()
        if not backend:
            continue
        if backend not in VALID_BACKENDS:
            valid = ", ".join((*VALID_BACKENDS, "all"))
            raise argparse.ArgumentTypeError(f"unknown backend {backend!r}; valid: {valid}")
        if backend not in selected:
            selected.append(backend)
    if not selected:
        raise argparse.ArgumentTypeError("at least one backend is required")
    return selected


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
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--seed",
        str(args.seed),
    ]
    if args.ignore_eos:
        command.append("--ignore-eos")
    if args.workload_jsonl:
        command.extend(["--workload-jsonl", str(Path(args.workload_jsonl).expanduser())])
    else:
        for prompt in args.prompts or DEFAULT_PROMPTS:
            command.extend(["--prompt", prompt])
    command.append("--json")
    command.append(str(Path(args.model_dir).expanduser()))
    return command


def parse_json_line(stdout: str, backend: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        data, _ = decoder.raw_decode(line)
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
    common = command_common_args(args)
    json_index = len(common) - 2
    common[json_index:json_index] = ["--max-num-batched-token-cap", str(args.max_num_batched_token_cap)]
    common[json_index:json_index] = [
        "--dtype",
        args.tinyllm_dtype,
        "--kv-cache-dtype",
        args.tinyllm_kv_cache_dtype,
    ]
    if args.profile_detail:
        common.insert(-1, "--profile-detail")
    if args.events_jsonl:
        common.insert(-1, str(Path(args.events_jsonl).expanduser()))
        common.insert(-2, "--events-jsonl")
    return run_command([str(binary), *common], "tinyllm")


def run_transformers(args: argparse.Namespace) -> Dict[str, Any]:
    script = Path(__file__).resolve().with_name("transformers_generate_benchmark.py")
    return run_command([sys.executable, str(script), *command_common_args(args)], "transformers")


def run_vllm(args: argparse.Namespace) -> Dict[str, Any]:
    script = Path(__file__).resolve().with_name("vllm_generate_benchmark.py")
    common = command_common_args(args)
    json_index = len(common) - 2
    vllm_args = [
        "--dtype",
        args.vllm_dtype,
        "--gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--max-model-len",
        str(args.vllm_max_model_len),
    ]
    if args.vllm_enforce_eager:
        vllm_args.append("--enforce-eager")
    common[json_index:json_index] = vllm_args
    return run_command([args.vllm_python, str(script), *common], "vllm")


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def add_pair_ratios(
    ratios: Dict[str, float],
    left_name: str,
    right_name: str,
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> None:
    prefix = f"{left_name}_vs_{right_name}"
    ratios[f"{prefix}_latency"] = ratio(float(left["avg_total_latency_ms"]), float(right["avg_total_latency_ms"]))
    ratios[f"{prefix}_first_token_latency"] = ratio(
        float(left["avg_first_token_latency_ms"]), float(right["avg_first_token_latency_ms"])
    )
    ratios[f"{prefix}_e2e_throughput"] = ratio(
        float(left["end_to_end_tokens_per_s"]), float(right["end_to_end_tokens_per_s"])
    )
    ratios[f"{prefix}_decode_throughput"] = ratio(
        float(left["decode_tokens_per_s"]), float(right["decode_tokens_per_s"])
    )
    ratios[f"{prefix}_load_init"] = ratio(float(left["avg_load_init_ms"]), float(right["avg_load_init_ms"]))


def target_generated_tokens(results: List[Dict[str, Any]]) -> int:
    if not results:
        return 0
    first = results[0]
    return int(first.get("prompt_count", 0)) * int(first.get("max_new_tokens", 0))


def generated_tokens_match(item: Dict[str, Any], target: int) -> bool:
    return abs(float(item.get("avg_generated_tokens", 0.0)) - float(target)) < 1e-6


def add_pair_comparison(
    ratios: Dict[str, float],
    ratio_status: Dict[str, Dict[str, Any]],
    left_name: str,
    right_name: str,
    left: Dict[str, Any],
    right: Dict[str, Any],
    target: int,
) -> None:
    prefix = f"{left_name}_vs_{right_name}"
    invalid = []
    if not generated_tokens_match(left, target):
        invalid.append(f"{left_name} generated {float(left.get('avg_generated_tokens', 0.0)):.3f}, expected {target}")
    if not generated_tokens_match(right, target):
        invalid.append(f"{right_name} generated {float(right.get('avg_generated_tokens', 0.0)):.3f}, expected {target}")
    ratio_status[prefix] = {
        "valid": not invalid,
        "reason": "; ".join(invalid),
        "target_generated_tokens": target,
    }
    if invalid:
        return
    add_pair_ratios(ratios, left_name, right_name, left, right)


def build_comparison(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_backend = {str(item["backend"]): item for item in results}
    target = target_generated_tokens(results)
    comparison: Dict[str, Any] = {
        "benchmark": "benchmark_comparison",
        "target_generated_tokens": target,
        "results": results,
        "ratios": {},
        "ratio_status": {},
    }
    tiny = by_backend.get("tinyllm")
    hf = by_backend.get("transformers")
    vllm = by_backend.get("vllm")
    ratios: Dict[str, float] = {}
    ratio_status: Dict[str, Dict[str, Any]] = {}
    if tiny is not None and hf is not None:
        add_pair_comparison(ratios, ratio_status, "tinyllm", "transformers", tiny, hf, target)
    if tiny is not None and vllm is not None:
        add_pair_comparison(ratios, ratio_status, "tinyllm", "vllm", tiny, vllm, target)
    if hf is not None and vllm is not None:
        add_pair_comparison(ratios, ratio_status, "transformers", "vllm", hf, vllm, target)
    comparison["ratios"] = ratios
    comparison["ratio_status"] = ratio_status
    return comparison


def print_ratio_group(ratios: Dict[str, Any], prefix: str, label: str) -> None:
    latency_key = f"{prefix}_latency"
    if latency_key not in ratios:
        return
    print(f"ratios {label}")
    print(f"  latency: {ratios[latency_key]:.3f}")
    print(f"  first_token_latency: {ratios[f'{prefix}_first_token_latency']:.3f}")
    print(f"  e2e_throughput: {ratios[f'{prefix}_e2e_throughput']:.3f}")
    print(f"  decode_throughput: {ratios[f'{prefix}_decode_throughput']:.3f}")
    print(f"  load_init: {ratios[f'{prefix}_load_init']:.3f}")


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
    for item in results:
        print(f"{item['backend']} details")
        print(f"  latency:")
        print(f"    load_init_ms: {float(item['avg_load_init_ms']):.3f}")
        print(f"    total_latency_ms: {float(item['avg_total_latency_ms']):.3f}")
        print(f"    first_token_latency_ms: {float(item['avg_first_token_latency_ms']):.3f}")
        if "prepare_inputs_ms" in item:
            print(f"    prepare_inputs_ms: {float(item['prepare_inputs_ms']):.3f}")
        if "prefill_ms" in item:
            print(f"    prefill_ms: {float(item['prefill_ms']):.3f}")
        if "decode_ms_total" in item:
            print(f"    decode_ms_total: {float(item['decode_ms_total']):.3f}")
        if "decode_ms_per_token" in item:
            print(f"    decode_ms_per_token: {float(item['decode_ms_per_token']):.3f}")
        if "sampling_ms" in item:
            print(f"    sampling_ms: {float(item['sampling_ms']):.3f}")
        detail_keys = [
            "embedding_ms",
            "qkv_proj_ms",
            "rope_ms",
            "attention_ms",
            "o_proj_ms",
            "mlp_ms",
            "norm_ms",
            "lm_head_ms",
        ]
        if any(key in item and float(item[key]) != 0.0 for key in detail_keys):
            print(f"  detailed_profile:")
            for key in detail_keys:
                if key in item:
                    print(f"    {key}: {float(item[key]):.3f}")
        print(f"  tokens:")
        print(f"    prompt_tokens: {int(item['prompt_tokens'])}")
        print(f"    avg_generated_tokens: {float(item['avg_generated_tokens']):.3f}")
        if "avg_decode_tokens" in item:
            print(f"    avg_decode_tokens: {float(item['avg_decode_tokens']):.3f}")
        print(f"  throughput:")
        print(f"    end_to_end_tokens_per_s: {float(item['end_to_end_tokens_per_s']):.3f}")
        print(f"    decode_tokens_per_s: {float(item['decode_tokens_per_s']):.3f}")

    ratios = comparison.get("ratios", {})
    ratio_status = comparison.get("ratio_status", {})
    for prefix, status in ratio_status.items():
        if status.get("valid", False):
            continue
        print(f"ratios {prefix}: invalid ({status.get('reason', '')})")
    if ratios:
        print_ratio_group(ratios, "tinyllm_vs_transformers", "tinyllm/transformers")
        print_ratio_group(ratios, "tinyllm_vs_vllm", "tinyllm/vllm")
        print_ratio_group(ratios, "transformers_vs_vllm", "transformers/vllm")


def main() -> int:
    args = parse_args()
    results: List[Dict[str, Any]] = []
    if "tinyllm" in args.selected_backends:
        results.append(run_tinyllm(args))
    if "transformers" in args.selected_backends:
        results.append(run_transformers(args))
    if "vllm" in args.selected_backends:
        results.append(run_vllm(args))
    comparison = build_comparison(results)
    comparison["benchmark_mode"] = args.benchmark_mode
    comparison["ignore_eos"] = bool(args.ignore_eos)
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
