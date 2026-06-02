#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_MODEL_DIR = "/models/Qwen2.5-1.5B-Instruct"
DEFAULT_TINYLLM_BINARY = "build-cuda/benchmark/llama_engine_benchmark"
DEFAULT_DEVICE = "cuda:0"
VALID_BACKENDS = ("tinyllm", "transformers", "vllm")

STANDARD_SCENARIOS = [
    {"name": "interactive", "batch": 1, "isl": 128, "osl": 64},
    {"name": "chat_serving", "batch": 8, "isl": 128, "osl": 128},
    {"name": "long_prefill", "batch": 4, "isl": 1024, "osl": 64},
    {"name": "decode_heavy", "batch": 4, "isl": 256, "osl": 256},
    {"name": "throughput", "batch": 16, "isl": 128, "osl": 128},
]

QUICK_SCENARIOS = [
    {"name": "quick_interactive", "batch": 1, "isl": 32, "osl": 8},
    {"name": "quick_batch", "batch": 2, "isl": 64, "osl": 8},
]

PROFILE_PREFILL_SCENARIOS = [
    {"name": "profile_prefill", "batch": 1, "isl": 64, "osl": 1},
]

PRESETS = {
    "quick": {
        "scenario_set": "quick",
        "quick": True,
        "backend": "all",
        "scenarios": "all",
        "transformers_scenarios": "all",
        "vllm_scenarios": "all",
        "warmup": 0,
        "repeat": 1,
        "label": "quick_validation",
    },
    "focus": {
        "scenario_set": "standard",
        "quick": False,
        "backend": "tinyllm",
        "scenarios": "interactive",
        "transformers_scenarios": "none",
        "vllm_scenarios": "none",
        "warmup": 0,
        "repeat": 1,
        "label": "focus_interactive",
    },
    "regression": {
        "scenario_set": "standard",
        "quick": False,
        "backend": "tinyllm",
        "scenarios": "interactive,chat_serving",
        "transformers_scenarios": "none",
        "vllm_scenarios": "none",
        "warmup": 1,
        "repeat": 1,
        "label": "regression_decode",
    },
    "full": {
        "scenario_set": "standard",
        "quick": False,
        "backend": "tinyllm",
        "scenarios": "all",
        "transformers_scenarios": "none",
        "vllm_scenarios": "none",
        "warmup": 1,
        "repeat": 3,
        "label": "full_after_optimization",
    },
    "profile_prefill": {
        "scenario_set": "profile_prefill",
        "quick": False,
        "backend": "tinyllm",
        "scenarios": "all",
        "transformers_scenarios": "none",
        "vllm_scenarios": "none",
        "warmup": 0,
        "repeat": 1,
        "label": "profile_prefill_detail",
    },
}

PROMPT_FRAGMENTS = [
    "You are evaluating an offline LLM inference engine. Discuss scheduler fairness, paged KV cache reuse, chunked prefill, decode throughput, tokenizer compatibility, CUDA execution, and failure handling with concrete engineering tradeoffs. ",
    "请用工程评审的方式分析一个大语言模型推理系统，覆盖连续批处理、KV 缓存、长上下文预填充、解码吞吐、显存压力、模型加载和线上稳定性。",
    "The benchmark should reflect production-like serving pressure: varied prompt lengths, batched requests, deterministic greedy decoding, repeatable measurements, and enough generated tokens to expose decode behavior. ",
    "在正式性能报告中，需要说明硬件、模型、输入输出 token 数、首 token 延迟、端到端吞吐、decode tokens/s、repeat 波动以及 baseline 对比。",
]


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


def parse_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_backend_selection(text: str) -> List[str]:
    if text == "all":
        return list(VALID_BACKENDS)
    selected: List[str] = []
    for item in parse_csv(text):
        if item not in VALID_BACKENDS:
            valid = ", ".join((*VALID_BACKENDS, "all"))
            raise argparse.ArgumentTypeError(f"unknown backend {item!r}; valid: {valid}")
        if item not in selected:
            selected.append(item)
    if not selected:
        raise argparse.ArgumentTypeError("at least one backend is required")
    return selected


def env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "") in {"1", "true", "TRUE", "on", "ON"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run industrial-style offline serving benchmark workloads")
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        help="named workload preset: quick, focus, regression, full, or profile_prefill",
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tinyllm-binary", default=DEFAULT_TINYLLM_BINARY)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--warmup", type=non_negative_int, default=1)
    parser.add_argument("--repeat", type=positive_int, default=3)
    parser.add_argument(
        "--benchmark-mode",
        choices=("correctness", "fixed_output_perf", "dual"),
        default="fixed_output_perf",
        help="correctness keeps EOS stops, fixed_output_perf ignores EOS, dual runs both",
    )
    parser.add_argument(
        "--backend",
        default="all",
        help="backend to run: all, tinyllm, transformers, vllm, or a comma-separated subset",
    )
    parser.add_argument("--scenarios", default="all", help="comma-separated scenario names, or all")
    parser.add_argument(
        "--transformers-scenarios",
        default="all",
        help="comma-separated scenarios that should include Transformers when --backend=all; use all or none",
    )
    parser.add_argument(
        "--vllm-scenarios",
        default="all",
        help="comma-separated scenarios that should include vLLM when --backend=all; use all or none",
    )
    parser.add_argument("--quick", action="store_true", help="run two short validation scenarios")
    parser.add_argument("--profile-detail", action="store_true", help="enable TinyLLM detailed runtime profiling")
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
    parser.add_argument("--output-dir", default="benchmark/results")
    parser.add_argument("--label", default="qwen25_1p5b_cuda4090")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    args = parser.parse_args()
    apply_preset(args)
    args.selected_backends = parse_backend_selection(args.backend)
    return args


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset is None:
        return
    preset = PRESETS[args.preset]
    for key in ("quick", "backend", "scenarios", "transformers_scenarios", "vllm_scenarios", "warmup", "repeat"):
        setattr(args, key, preset[key])
    args.scenario_set = str(preset.get("scenario_set", "quick" if args.quick else "standard"))
    if args.label == "qwen25_1p5b_cuda4090":
        args.label = str(preset["label"])


def import_tokenizer(model_dir: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(f"missing transformers dependency for prompt generation: {exc}") from exc
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)


def token_count(tokenizer: Any, text: str) -> int:
    return int(len(tokenizer.encode(text, add_special_tokens=True)))


def make_prompt(tokenizer: Any, target_tokens: int, seed: int) -> Tuple[str, int]:
    fragment_order = [PROMPT_FRAGMENTS[(seed + idx) % len(PROMPT_FRAGMENTS)] for idx in range(len(PROMPT_FRAGMENTS))]
    text = f"Request {seed}: "
    idx = 0
    while token_count(tokenizer, text) < target_tokens:
        text += fragment_order[idx % len(fragment_order)]
        idx += 1

    lo = 1
    hi = len(text)
    best = text
    best_count = token_count(tokenizer, best)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        count = token_count(tokenizer, candidate)
        if count <= target_tokens:
            best = candidate
            best_count = count
            lo = mid + 1
        else:
            hi = mid - 1

    if best_count < max(1, target_tokens - 2):
        # Some tokenizers make exact targeting difficult around UTF-8 boundaries; prefer a stable prompt near target.
        best = text[: min(len(text), len(best) + 64)]
        best_count = token_count(tokenizer, best)
    return best, best_count


def select_scenarios(args: argparse.Namespace) -> List[Dict[str, int]]:
    scenario_set = getattr(args, "scenario_set", "quick" if args.quick else "standard")
    if scenario_set == "profile_prefill":
        scenarios = PROFILE_PREFILL_SCENARIOS
    elif args.quick or scenario_set == "quick":
        scenarios = QUICK_SCENARIOS
    else:
        scenarios = STANDARD_SCENARIOS
    if args.scenarios == "all":
        return list(scenarios)
    wanted = set(parse_csv(args.scenarios))
    selected = [item for item in scenarios if str(item["name"]) in wanted]
    missing = wanted - {str(item["name"]) for item in selected}
    if missing:
        valid = ", ".join(str(item["name"]) for item in scenarios)
        raise RuntimeError(f"unknown scenario(s): {', '.join(sorted(missing))}; valid: {valid}")
    return selected


def transformers_enabled_for(args: argparse.Namespace, scenario_name: str) -> bool:
    if "transformers" not in args.selected_backends:
        return False
    if args.transformers_scenarios == "all":
        return True
    if args.transformers_scenarios == "none":
        return False
    return scenario_name in set(parse_csv(args.transformers_scenarios))


def vllm_enabled_for(args: argparse.Namespace, scenario_name: str) -> bool:
    if "vllm" not in args.selected_backends:
        return False
    if args.vllm_scenarios == "all":
        return True
    if args.vllm_scenarios == "none":
        return False
    return scenario_name in set(parse_csv(args.vllm_scenarios))


def enabled_backends_for(args: argparse.Namespace, scenario_name: str) -> Tuple[List[str], List[str]]:
    enabled = []
    omitted = []
    for backend in args.selected_backends:
        include = True
        if backend == "transformers":
            include = transformers_enabled_for(args, scenario_name)
        elif backend == "vllm":
            include = vllm_enabled_for(args, scenario_name)
        if include:
            enabled.append(backend)
        else:
            omitted.append(backend)
    if not enabled:
        raise RuntimeError(f"all selected backends were omitted for scenario {scenario_name}")
    return enabled, omitted


def benchmark_runs_for(args: argparse.Namespace, scenario_name: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if args.benchmark_mode in {"correctness", "dual"}:
        correctness_backends = [backend for backend in args.selected_backends if backend in {"tinyllm", "transformers"}]
        if correctness_backends:
            omitted = [backend for backend in args.selected_backends if backend not in correctness_backends]
            runs.append(
                {
                    "mode": "correctness",
                    "enabled": correctness_backends,
                    "omitted": omitted,
                    "warmup": 0,
                    "repeat": 1,
                    "ignore_eos": False,
                }
            )
    if args.benchmark_mode in {"fixed_output_perf", "dual"}:
        enabled, omitted = enabled_backends_for(args, scenario_name)
        runs.append(
            {
                "mode": "fixed_output_perf",
                "enabled": enabled,
                "omitted": omitted,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "ignore_eos": True,
            }
        )
    if not runs:
        raise RuntimeError(f"no benchmark runs are enabled for scenario {scenario_name}")
    return runs


def vllm_model_len_for(args: argparse.Namespace, scenario: Dict[str, int]) -> int:
    if args.vllm_max_model_len == 0:
        return 0
    required = int(scenario["isl"]) + int(scenario["osl"])
    return max(args.vllm_max_model_len, required, 256)


def command_for(
    args: argparse.Namespace,
    scenario: Dict[str, int],
    prompts: Sequence[str],
    backend: str,
    benchmark_mode: str,
    warmup: int,
    repeat: int,
    ignore_eos: bool,
) -> List[str]:
    command = [
        sys.executable,
        "benchmark/run_benchmark_comparison.py",
        "--tinyllm-binary",
        args.tinyllm_binary,
        "--backend",
        backend,
        "--device",
        args.device,
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--max-new-tokens",
        str(scenario["osl"]),
        "--benchmark-mode",
        benchmark_mode,
        "--max-num-batched-token-cap",
        str(args.max_num_batched_token_cap),
        "--vllm-python",
        args.vllm_python,
        "--vllm-dtype",
        args.vllm_dtype,
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm-max-model-len",
        str(vllm_model_len_for(args, scenario)),
    ]
    if ignore_eos:
        command.append("--ignore-eos")
    if args.profile_detail:
        command.append("--profile-detail")
    if args.vllm_enforce_eager:
        command.append("--vllm-enforce-eager")
    for prompt in prompts:
        command.extend(["--prompt", prompt])
    command.extend(["--json", args.model_dir])
    return command


def parse_json_line(stdout: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            data, _ = decoder.raw_decode(line)
            return data
    raise RuntimeError("benchmark command did not emit a final JSON line")


def run_command(command: Sequence[str], dry_run: bool) -> Dict[str, Any]:
    if dry_run:
        return {"dry_run_command": list(command)}
    run = subprocess.run(command, check=False, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(
            "benchmark command failed with exit code "
            f"{run.returncode}: {' '.join(command[:8])}...\nSTDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}"
        )
    return parse_json_line(run.stdout)


def query_gpu() -> Dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    try:
        run = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception as exc:
        return {"error": str(exc)}
    line = run.stdout.strip().splitlines()[0] if run.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    keys = ["name", "memory_total", "driver_version"]
    return {key: parts[idx] for idx, key in enumerate(keys) if idx < len(parts)}


def flatten_results(comparison: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "results" in comparison:
        return list(comparison["results"])
    return [comparison]


def fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_fence(text: Any) -> str:
    value = "" if text is None else str(text)
    longest = 0
    current = 0
    for ch in value:
        if ch == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{value}\n{fence}"


def samples_by_prompt(results: Sequence[Dict[str, Any]], prompt_index: int) -> List[Tuple[str, Dict[str, Any]]]:
    samples = []
    for result in results:
        backend = str(result.get("backend", "unknown"))
        backend_samples = result.get("samples", [])
        if isinstance(backend_samples, list) and prompt_index < len(backend_samples):
            sample = backend_samples[prompt_index]
            if isinstance(sample, dict):
                samples.append((backend, sample))
    return samples


def sample_generated_text(sample: Dict[str, Any]) -> str:
    return str(sample.get("generated_text", sample.get("output_text", "")))


def sample_token_ids(sample: Dict[str, Any]) -> List[int]:
    token_ids = sample.get("token_ids", [])
    if not isinstance(token_ids, list):
        return []
    return [int(token_id) for token_id in token_ids]


def build_output_agreement(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    max_prompts = 0
    for result in results:
        samples = result.get("samples", [])
        if isinstance(samples, list):
            max_prompts = max(max_prompts, len(samples))

    agreements = []
    for prompt_index in range(max_prompts):
        prompt_samples = samples_by_prompt(results, prompt_index)
        if len(prompt_samples) < 2:
            continue
        reference_backend, reference_sample = prompt_samples[0]
        reference_text = sample_generated_text(reference_sample)
        reference_token_ids = sample_token_ids(reference_sample)
        for backend, sample in prompt_samples[1:]:
            token_ids = sample_token_ids(sample)
            agreements.append(
                {
                    "prompt_index": prompt_index,
                    "reference_backend": reference_backend,
                    "backend": backend,
                    "text_match": sample_generated_text(sample) == reference_text,
                    "token_ids_match": token_ids == reference_token_ids,
                    "reference_token_ids": reference_token_ids,
                    "backend_token_ids": token_ids,
                }
            )
    return agreements


def ratio_row(ratios: Dict[str, Any], prefix: str, label: str) -> Optional[Tuple[str, str, str, str, str, str]]:
    latency_key = f"{prefix}_latency"
    if latency_key not in ratios:
        return None
    return (
        label,
        fmt_float(ratios.get(latency_key)),
        fmt_float(ratios.get(f"{prefix}_first_token_latency")),
        fmt_float(ratios.get(f"{prefix}_e2e_throughput")),
        fmt_float(ratios.get(f"{prefix}_decode_throughput")),
        fmt_float(ratios.get(f"{prefix}_load_init")),
    )


def build_markdown(report: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Industrial Benchmark Report: {report['label']}")
    lines.append("")
    lines.append(f"- generated_at: `{report['generated_at']}`")
    lines.append(f"- model: `{report['model_dir']}`")
    lines.append(f"- device: `{report['device']}`")
    gpu = report.get("gpu", {})
    if gpu:
        lines.append(f"- gpu: `{gpu.get('name', 'unknown')}`, memory `{gpu.get('memory_total', 'unknown')}`")
    if report.get("preset"):
        lines.append(f"- preset: `{report['preset']}`")
    lines.append(f"- benchmark_mode: `{report.get('benchmark_mode', '-')}`")
    lines.append(f"- warmup/repeat: `{report['warmup']}/{report['repeat']}`")
    lines.append(f"- profile_detail: `{'on' if report.get('profile_detail') else 'off'}`")
    lines.append("")
    lines.append("Current benchmark mode is offline batched generation. It is not a request-rate server benchmark with p50/p95/p99 latency.")
    lines.append("Sample output is reported for sanity; exact generated-token agreement is tracked separately below.")
    lines.append("")
    lines.append("| scenario | mode | backend | batch | ISL target | prompt tokens | OSL | target gen | generated | TTFT ms | latency ms | decode ms/token | e2e tok/s | decode tok/s |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for scenario in report["scenarios"]:
        meta = scenario["workload"]
        for result in scenario.get("results", []):
            lines.append(
                "| {scenario} | {mode} | {backend} | {batch} | {isl} | {prompt_tokens} | {osl} | {target_generated} | {generated} | {ttft} | {latency} | {decode_ms} | {e2e} | {decode_tps} |".format(
                    scenario=meta["name"],
                    mode=scenario.get("benchmark_mode", "-"),
                    backend=result.get("backend", "-"),
                    batch=meta["batch"],
                    isl=meta["isl"],
                    prompt_tokens=result.get("prompt_tokens", scenario.get("actual_prompt_tokens", "-")),
                    osl=meta["osl"],
                    target_generated=scenario.get("target_generated_tokens", "-"),
                    generated=fmt_float(result.get("avg_generated_tokens")),
                    ttft=fmt_float(result.get("avg_first_token_latency_ms")),
                    latency=fmt_float(result.get("avg_total_latency_ms")),
                    decode_ms=fmt_float(result.get("decode_ms_per_token")),
                    e2e=fmt_float(result.get("end_to_end_tokens_per_s")),
                    decode_tps=fmt_float(result.get("decode_tokens_per_s")),
                )
            )
        if scenario.get("omitted_baselines"):
            lines.append(
                f"| {meta['name']} | {scenario.get('benchmark_mode', '-')} | omitted: {', '.join(scenario['omitted_baselines'])} | {meta['batch']} | {meta['isl']} | {scenario.get('actual_prompt_tokens', '-')} | {meta['osl']} | {scenario.get('target_generated_tokens', '-')} | - | - | - | - | - | - |"
            )
    lines.append("")
    lines.append("## Output Agreement")
    lines.append("")
    lines.append("Each row compares a backend to the first backend emitted for that scenario, usually TinyLLM.")
    lines.append("")
    lines.append("| scenario | mode | prompt | reference | backend | text match | token ids match |")
    lines.append("| --- | --- | ---: | --- | --- | --- | --- |")
    for scenario in report["scenarios"]:
        for agreement in scenario.get("output_agreement", []):
            lines.append(
                "| {scenario} | {mode} | {prompt} | {reference} | {backend} | {text_match} | {token_match} |".format(
                    scenario=scenario["workload"]["name"],
                    mode=scenario.get("benchmark_mode", "-"),
                    prompt=agreement.get("prompt_index", "-"),
                    reference=agreement.get("reference_backend", "-"),
                    backend=agreement.get("backend", "-"),
                    text_match="yes" if agreement.get("text_match") else "no",
                    token_match="yes" if agreement.get("token_ids_match") else "no",
                )
            )
    lines.append("")
    lines.append("## Ratio Validity")
    lines.append("")
    lines.append("Ratios are valid only when both compared backends generated the target number of tokens.")
    lines.append("")
    lines.append("| scenario | mode | ratio | valid | target generated | reason |")
    lines.append("| --- | --- | --- | --- | ---: | --- |")
    for scenario in report["scenarios"]:
        for prefix, status in scenario.get("ratio_status", {}).items():
            lines.append(
                "| {scenario} | {mode} | {ratio} | {valid} | {target} | {reason} |".format(
                    scenario=scenario["workload"]["name"],
                    mode=scenario.get("benchmark_mode", "-"),
                    ratio=prefix,
                    valid="yes" if status.get("valid") else "no",
                    target=status.get("target_generated_tokens", "-"),
                    reason=status.get("reason", ""),
                )
            )
    lines.append("")
    lines.append("## Backend Ratios")
    lines.append("")
    lines.append("| scenario | mode | ratio | latency | TTFT | e2e throughput | decode throughput | load/init |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for scenario in report["scenarios"]:
        ratios = scenario.get("ratios", {})
        if not ratios:
            continue
        rows = [
            ratio_row(ratios, "tinyllm_vs_transformers", "TinyLLM / Transformers"),
            ratio_row(ratios, "tinyllm_vs_vllm", "TinyLLM / vLLM"),
            ratio_row(ratios, "transformers_vs_vllm", "Transformers / vLLM"),
        ]
        for row in rows:
            if row is None:
                continue
            label, latency, ttft, e2e, decode, load = row
            lines.append(
                "| {scenario} | {mode} | {label} | {latency} | {ttft} | {e2e} | {decode} | {load} |".format(
                    scenario=scenario["workload"]["name"],
                    mode=scenario.get("benchmark_mode", "-"),
                    label=label,
                    latency=latency,
                    ttft=ttft,
                    e2e=e2e,
                    decode=decode,
                    load=load,
                )
            )
    lines.append("")
    lines.append("## Prompts And Outputs")
    lines.append("")
    for scenario in report["scenarios"]:
        meta = scenario["workload"]
        lines.append(f"### {meta['name']} ({scenario.get('benchmark_mode', '-')})")
        prompts = scenario.get("prompts", [])
        per_prompt_tokens = scenario.get("per_prompt_tokens", [])
        for prompt_index, prompt in enumerate(prompts):
            token_count_text = ""
            if prompt_index < len(per_prompt_tokens):
                token_count_text = f" ({per_prompt_tokens[prompt_index]} tokens)"
            lines.append("")
            lines.append(f"#### Prompt {prompt_index}{token_count_text}")
            lines.append(markdown_fence(prompt))
            for backend, sample in samples_by_prompt(scenario.get("results", []), prompt_index):
                finish_reason = sample.get("finish_reason", "")
                token_ids = sample.get("token_ids", [])
                token_count = len(token_ids) if isinstance(token_ids, list) else "-"
                suffix = f", finish={finish_reason}" if finish_reason else ""
                lines.append("")
                lines.append(f"##### {backend} output ({token_count} tokens{suffix})")
                lines.append(markdown_fence(sample.get("generated_text", sample.get("output_text", ""))))
        lines.append("")

    if report.get("profile_detail"):
        lines.append("")
        lines.append("## TinyLLM Detailed Profile")
        lines.append("")
        lines.append("These fields include extra synchronization and are for bottleneck diagnosis, not headline throughput.")
        lines.append("")
        lines.append("| scenario | embedding | qkv proj | rope | attention | o proj | mlp | norm | lm head |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for scenario in report["scenarios"]:
            for result in scenario.get("results", []):
                if result.get("backend") != "tinyllm":
                    continue
                lines.append(
                    "| {scenario} | {embedding} | {qkv} | {rope} | {attention} | {oproj} | {mlp} | {norm} | {lm_head} |".format(
                        scenario=scenario["workload"]["name"],
                        embedding=fmt_float(result.get("embedding_ms")),
                        qkv=fmt_float(result.get("qkv_proj_ms")),
                        rope=fmt_float(result.get("rope_ms")),
                        attention=fmt_float(result.get("attention_ms")),
                        oproj=fmt_float(result.get("o_proj_ms")),
                        mlp=fmt_float(result.get("mlp_ms")),
                        norm=fmt_float(result.get("norm_ms")),
                        lm_head=fmt_float(result.get("lm_head_ms")),
                    )
                )
    return "\n".join(lines) + "\n"


def write_reports(args: argparse.Namespace, report: Dict[str, Any]) -> Tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = report["generated_at"].replace(":", "").replace("-", "").replace("T", "_").split("+")[0]
    stem = f"{args.label}_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")

    keep = {json_path.resolve(), md_path.resolve()}
    for path in output_dir.iterdir():
        if path.resolve() in keep or not path.is_file():
            continue
        if path.suffix in {".json", ".md"}:
            path.unlink()
    return json_path, md_path


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise RuntimeError(f"model directory does not exist: {model_dir}")
    tokenizer = import_tokenizer(model_dir)
    scenarios = select_scenarios(args)

    report: Dict[str, Any] = {
        "label": args.label,
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "device": args.device,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "benchmark_mode": args.benchmark_mode,
        "backend": args.backend,
        "selected_backends": args.selected_backends,
        "preset": args.preset,
        "profile_detail": args.profile_detail or env_flag_enabled("TINYLLM_PROFILE_DETAIL"),
        "gpu": query_gpu(),
        "vllm": {
            "python": args.vllm_python,
            "dtype": args.vllm_dtype,
            "max_model_len": args.vllm_max_model_len,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "enforce_eager": args.vllm_enforce_eager,
        },
        "scenarios": [],
    }

    for scenario in scenarios:
        prompts = []
        per_prompt_tokens = []
        for idx in range(int(scenario["batch"])):
            prompt, count = make_prompt(tokenizer, int(scenario["isl"]), idx)
            prompts.append(prompt)
            per_prompt_tokens.append(count)

        for run in benchmark_runs_for(args, str(scenario["name"])):
            scenario_backend = ",".join(run["enabled"])
            command = command_for(
                args,
                scenario,
                prompts,
                scenario_backend,
                str(run["mode"]),
                int(run["warmup"]),
                int(run["repeat"]),
                bool(run["ignore_eos"]),
            )
            print(
                f"running {scenario['name']} [{run['mode']}]: backend={scenario_backend}, "
                f"batch={scenario['batch']}, target_isl={scenario['isl']}, "
                f"actual_prompt_tokens={sum(per_prompt_tokens)}, osl={scenario['osl']}",
                flush=True,
            )
            comparison = run_command(command, args.dry_run)
            results = flatten_results(comparison)
            report["scenarios"].append(
                {
                    "workload": scenario,
                    "benchmark_mode": str(run["mode"]),
                    "ignore_eos": bool(run["ignore_eos"]),
                    "warmup": int(run["warmup"]),
                    "repeat": int(run["repeat"]),
                    "actual_prompt_tokens": sum(per_prompt_tokens),
                    "per_prompt_tokens": per_prompt_tokens,
                    "target_generated_tokens": comparison.get(
                        "target_generated_tokens",
                        int(scenario["batch"]) * int(scenario["osl"]),
                    ),
                    "command": command,
                    "prompts": list(prompts),
                    "results": results,
                    "ratios": comparison.get("ratios", {}),
                    "ratio_status": comparison.get("ratio_status", {}),
                    "output_agreement": build_output_agreement(results),
                    "omitted_baselines": run["omitted"],
                }
            )

    json_path, md_path = write_reports(args, report)
    print(f"wrote JSON report: {json_path}")
    print(f"wrote Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"industrial_benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
