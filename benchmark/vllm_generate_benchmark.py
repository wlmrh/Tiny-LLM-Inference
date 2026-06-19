#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive float")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vLLM offline greedy generation")
    parser.add_argument("--device", default="cuda:0", help="cpu, cuda, or cuda:<device_id>")
    parser.add_argument("--warmup", type=non_negative_int, default=1)
    parser.add_argument("--repeat", type=positive_int, default=3)
    parser.add_argument("--max-new-tokens", type=positive_int, default=8)
    parser.add_argument("--ignore-eos", action="store_true", help="generate exactly max_new_tokens unless another hard limit is hit")
    parser.add_argument("--prompt", action="append", dest="prompts", default=[])
    parser.add_argument("--workload-jsonl", help="flat JSONL workload with prompt/request_id records")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=non_negative_int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto", help="vLLM dtype, for example auto, float16, bfloat16, or float32")
    parser.add_argument("--gpu-memory-utilization", type=positive_float, default=0.90)
    parser.add_argument(
        "--max-model-len",
        type=non_negative_int,
        default=4096,
        help="vLLM max_model_len; use 0 to leave the model config default unchanged",
    )
    parser.add_argument("--enforce-eager", action="store_true", help="disable vLLM CUDA graph capture")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("model_dir")
    return parser.parse_args()


def load_workload_jsonl(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    if not args.workload_jsonl:
        prompts = args.prompts or DEFAULT_PROMPTS
        return prompts, [f"request-{idx}" for idx in range(len(prompts))]

    prompts: List[str] = []
    request_ids: List[str] = []
    max_new_tokens: Optional[int] = None
    ignore_eos: Optional[bool] = None
    with Path(args.workload_jsonl).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if not isinstance(prompt, str):
                raise RuntimeError(f"workload record {line_number} missing string prompt")
            prompts.append(prompt)
            request_ids.append(str(record.get("request_id", f"request-{len(request_ids)}")))
            if "max_new_tokens" in record:
                value = int(record["max_new_tokens"])
                if max_new_tokens is not None and max_new_tokens != value:
                    raise RuntimeError("mixed workload max_new_tokens values are not supported")
                max_new_tokens = value
            if "ignore_eos" in record:
                value = bool(record["ignore_eos"])
                if ignore_eos is not None and ignore_eos != value:
                    raise RuntimeError("mixed workload ignore_eos values are not supported")
                ignore_eos = value
            for key, attr, caster in (
                ("temperature", "temperature", float),
                ("top_p", "top_p", float),
                ("top_k", "top_k", int),
                ("repetition_penalty", "repetition_penalty", float),
            ):
                if key in record:
                    current = getattr(args, attr)
                    value = caster(record[key])
                    if prompts[:-1] and current != value:
                        raise RuntimeError(f"mixed workload {key} values are not supported")
                    setattr(args, attr, value)
    if not prompts:
        raise RuntimeError(f"workload JSONL is empty: {args.workload_jsonl}")
    if max_new_tokens is not None:
        args.max_new_tokens = max_new_tokens
    if ignore_eos is not None:
        args.ignore_eos = ignore_eos
    return prompts, request_ids


def validate_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise RuntimeError(f"model_dir is not a directory: {model_dir}")
    if not (model_dir / "config.json").exists():
        raise RuntimeError(f"model_dir must contain config.json: {model_dir}")
    if not (model_dir / "tokenizer.json").exists() and not (model_dir / "tokenizer.model").exists():
        raise RuntimeError(f"model_dir must contain tokenizer.json or tokenizer.model: {model_dir}")
    if not (model_dir / "model.safetensors").exists() and not any(model_dir.glob("*.safetensors")):
        raise RuntimeError(f"model_dir must contain model.safetensors or safetensors shards: {model_dir}")


def configure_device(device_text: str) -> str:
    if device_text == "cpu":
        return "cpu"
    if device_text == "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        return "cuda"
    if device_text.startswith("cuda:"):
        suffix = device_text.split(":", 1)[1]
        if not suffix.isdigit():
            raise RuntimeError("device must be cpu, cuda, or cuda:<device_id>")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(suffix))
        return "cuda"
    raise RuntimeError("device must be cpu, cuda, or cuda:<device_id>")


def import_deps():
    try:
        import torch
        import vllm
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(f"missing Python dependency for vLLM benchmark: {exc}") from exc
    return torch, vllm, AutoTokenizer, LLM, SamplingParams


def validate_runtime_device(device_kind: str, torch) -> None:
    if device_kind != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("--device cuda requires torch.cuda.is_available()")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("no CUDA device is visible to vLLM")


def make_synchronizer(device_kind: str, torch):
    def synchronize() -> None:
        if device_kind == "cuda":
            torch.cuda.synchronize()

    return synchronize


def count_prompt_tokens(tokenizer, prompts: List[str]) -> int:
    total = 0
    for prompt in prompts:
        total += int(len(tokenizer.encode(prompt, add_special_tokens=True)))
    return total


def load_engine(args: argparse.Namespace, model_dir: Path, device_kind: str, llm_cls, torch) -> Tuple[Any, float]:
    load_start = time.perf_counter()
    kwargs: Dict[str, Any] = {
        "model": str(model_dir),
        "tokenizer": str(model_dir),
        "trust_remote_code": False,
        "dtype": args.dtype,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": bool(args.enforce_eager),
    }
    if args.max_model_len > 0:
        kwargs["max_model_len"] = args.max_model_len
    if device_kind == "cpu":
        kwargs["device"] = "cpu"
    try:
        llm = llm_cls(**kwargs)
    except TypeError:
        if "device" not in kwargs:
            raise
        kwargs.pop("device")
        llm = llm_cls(**kwargs)
    if device_kind == "cuda":
        torch.cuda.synchronize()
    load_ms = (time.perf_counter() - load_start) * 1000.0
    return llm, load_ms


def output_metrics_first_token_ms(outputs: List[Any]) -> Optional[float]:
    first_token_ms: List[float] = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        if metrics is None:
            continue
        first_time = getattr(metrics, "first_token_time", None)
        arrival_time = getattr(metrics, "arrival_time", None)
        if first_time is None or arrival_time is None:
            continue
        delta = float(first_time) - float(arrival_time)
        if delta >= 0.0:
            first_token_ms.append(delta * 1000.0)
    if not first_token_ms:
        return None
    return min(first_token_ms)


def extract_samples(outputs: List[Any], request_ids: List[str]) -> Tuple[int, List[Dict[str, Any]]]:
    generated_tokens = 0
    samples = []
    for idx, request_output in enumerate(outputs):
        prompt = str(getattr(request_output, "prompt", ""))
        completions = list(getattr(request_output, "outputs", []))
        completion = completions[0] if completions else None
        token_ids = list(getattr(completion, "token_ids", [])) if completion is not None else []
        text = str(getattr(completion, "text", "")) if completion is not None else ""
        finish_reason = getattr(completion, "finish_reason", "") if completion is not None else ""
        generated_tokens += len(token_ids)
        samples.append(
            {
                "request_id": request_ids[idx] if idx < len(request_ids) else f"request-{idx}",
                "prompt": prompt,
                "output_text": text,
                "generated_text": text,
                "token_ids": [int(token_id) for token_id in token_ids],
                "finished": bool(finish_reason),
                "finish_reason": str(finish_reason),
            }
        )
    return generated_tokens, samples


def generate_once(
    llm,
    sampling_params,
    first_token_sampling_params,
    prompts: List[str],
    request_ids: List[str],
    prompt_tokens: int,
    load_ms: float,
    device_kind: str,
    torch,
    measure: bool,
) -> Optional[Dict[str, Any]]:
    synchronize = make_synchronizer(device_kind, torch)

    synchronize()
    generation_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    synchronize()
    generation_end = time.perf_counter()

    if not measure:
        return None

    first_token_ms = output_metrics_first_token_ms(outputs)
    ttft_source = "vllm_metrics"
    if first_token_ms is None:
        synchronize()
        probe_start = time.perf_counter()
        llm.generate(prompts, first_token_sampling_params, use_tqdm=False)
        synchronize()
        first_token_ms = (time.perf_counter() - probe_start) * 1000.0
        ttft_source = "one_token_probe"

    generated_tokens, samples = extract_samples(outputs, request_ids)
    return {
        "load_ms": load_ms,
        "total_ms": (generation_end - generation_start) * 1000.0,
        "first_token_ms": first_token_ms,
        "ttft_source": ttft_source,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "samples": samples,
    }


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_summary(
    args: argparse.Namespace,
    model_dir: Path,
    device_text: str,
    prompts: List[str],
    repeats: List[Dict[str, Any]],
    vllm_version: str,
) -> Dict[str, Any]:
    load_ms = [float(item["load_ms"]) for item in repeats]
    total_ms = [float(item["total_ms"]) for item in repeats]
    first_ms = [float(item["first_token_ms"]) for item in repeats]
    total_generated_tokens = sum(int(item["generated_tokens"]) for item in repeats)
    prompt_tokens = int(repeats[0]["prompt_tokens"]) if repeats else 0
    avg_generated_tokens = total_generated_tokens / len(repeats) if repeats else 0.0
    avg_total_ms = mean(total_ms)
    avg_first_ms = mean(first_ms)
    e2e_tokens_per_s = avg_generated_tokens / (avg_total_ms / 1000.0) if avg_total_ms > 0.0 else 0.0
    decode_ms = max(0.0, avg_total_ms - avg_first_ms)
    decode_tokens = max(0.0, avg_generated_tokens - float(len(prompts)))
    decode_tokens_per_s = decode_tokens / (decode_ms / 1000.0) if decode_ms > 0.0 else 0.0
    decode_ms_per_token = decode_ms / decode_tokens if decode_tokens > 0.0 else 0.0
    ttft_sources = sorted({str(item.get("ttft_source", "")) for item in repeats if item.get("ttft_source")})
    return {
        "benchmark": "vllm_generate_benchmark",
        "backend": "vllm",
        "model": str(model_dir),
        "device": device_text,
        "prompt_count": len(prompts),
        "prompt_tokens": prompt_tokens,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "max_new_tokens": args.max_new_tokens,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "repetition_penalty": float(args.repetition_penalty),
        "seed": int(args.seed),
        "ignore_eos": bool(args.ignore_eos),
        "dtype": args.dtype,
        "vllm_version": vllm_version,
        "vllm_max_model_len": args.max_model_len,
        "vllm_gpu_memory_utilization": args.gpu_memory_utilization,
        "vllm_enforce_eager": bool(args.enforce_eager),
        "ttft_source": ",".join(ttft_sources),
        "avg_load_init_ms": mean(load_ms),
        "avg_total_latency_ms": avg_total_ms,
        "avg_first_token_latency_ms": avg_first_ms,
        "avg_generated_tokens": avg_generated_tokens,
        "total_generated_tokens": total_generated_tokens,
        "decode_ms_total": decode_ms,
        "decode_ms_per_token": decode_ms_per_token,
        "avg_decode_tokens": decode_tokens,
        "end_to_end_tokens_per_s": e2e_tokens_per_s,
        "decode_tokens_per_s": decode_tokens_per_s,
        "repeat_total_latency_ms": total_ms,
        "repeat_load_init_ms": load_ms,
        "samples": repeats[0].get("samples", []) if repeats else [],
    }


def print_summary(summary: Dict[str, Any], emit_json: bool) -> None:
    print("vllm_generate_benchmark")
    print(f"  model: {summary['model']}")
    print(f"  device: {summary['device']}")
    print(
        f"  prompts: {summary['prompt_count']}, warmup: {summary['warmup']}, "
        f"repeat: {summary['repeat']}, max_new_tokens: {summary['max_new_tokens']}, "
        f"ignore_eos: {'on' if summary['ignore_eos'] else 'off'}, "
        f"temperature: {summary['temperature']}, top_p: {summary['top_p']}, top_k: {summary['top_k']}"
    )
    print(
        f"  vllm: version {summary['vllm_version']}, dtype {summary['dtype']}, "
        f"max_model_len {summary['vllm_max_model_len']}, "
        f"gpu_memory_utilization {summary['vllm_gpu_memory_utilization']:.3f}"
    )
    print("  latency:")
    print(f"    avg_load_init_ms: {summary['avg_load_init_ms']:.3f}")
    print(f"    avg_total_latency_ms: {summary['avg_total_latency_ms']:.3f}")
    print(f"    avg_first_token_latency_ms: {summary['avg_first_token_latency_ms']:.3f}")
    print(f"    ttft_source: {summary['ttft_source']}")
    print(f"    decode_ms_total: {summary['decode_ms_total']:.3f}")
    print(f"    decode_ms_per_token: {summary['decode_ms_per_token']:.3f}")
    print("  tokens:")
    print(f"    prompt_tokens: {summary['prompt_tokens']}")
    print(f"    avg_generated_tokens: {summary['avg_generated_tokens']:.3f}")
    print(f"    total_generated_tokens: {summary['total_generated_tokens']}")
    print(f"    avg_decode_tokens: {summary['avg_decode_tokens']:.3f}")
    print("  throughput:")
    print(f"    end_to_end_tokens_per_s: {summary['end_to_end_tokens_per_s']:.3f}")
    print(f"    decode_tokens_per_s: {summary['decode_tokens_per_s']:.3f}")
    print("  repeats:")
    print(f"    repeat_total_latency_ms: {[round(x, 3) for x in summary['repeat_total_latency_ms']]}")
    print(f"    repeat_load_init_ms: {[round(x, 3) for x in summary['repeat_load_init_ms']]}")
    if summary.get("samples"):
        print("  samples:")
        for idx, sample in enumerate(summary["samples"]):
            print(f"    [{idx}] prompt: {sample['prompt']}")
            print(f"    [{idx}] output_text: {sample['output_text']}")
            print(f"    [{idx}] generated_text: {sample.get('generated_text', sample['output_text'])}")
            print(f"    [{idx}] finish_reason: {sample['finish_reason']}")
    if emit_json:
        print("  json: see final machine-readable line below")
        print(json.dumps(summary, separators=(",", ":")))


def main() -> int:
    args = parse_args()
    prompts, request_ids = load_workload_jsonl(args)
    model_dir = Path(args.model_dir).expanduser()
    validate_model_dir(model_dir)
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    device_kind = configure_device(args.device)
    torch, vllm, auto_tokenizer, llm_cls, sampling_params_cls = import_deps()
    validate_runtime_device(device_kind, torch)

    tokenizer = auto_tokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    prompt_tokens = count_prompt_tokens(tokenizer, prompts)
    llm, load_ms = load_engine(args, model_dir, device_kind, llm_cls, torch)
    if args.seed:
        torch.manual_seed(int(args.seed))
    sampling_kwargs = {
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "repetition_penalty": float(args.repetition_penalty),
        "ignore_eos": bool(args.ignore_eos),
    }
    if int(args.top_k) > 0:
        sampling_kwargs["top_k"] = int(args.top_k)
    sampling_params = sampling_params_cls(max_tokens=args.max_new_tokens, **sampling_kwargs)
    first_token_sampling_params = sampling_params_cls(max_tokens=1, **sampling_kwargs)

    for _ in range(args.warmup):
        generate_once(
            llm,
            sampling_params,
            first_token_sampling_params,
            prompts,
            request_ids,
            prompt_tokens,
            load_ms,
            device_kind,
            torch,
            False,
        )

    repeats = []
    for _ in range(args.repeat):
        metrics = generate_once(
            llm,
            sampling_params,
            first_token_sampling_params,
            prompts,
            request_ids,
            prompt_tokens,
            load_ms,
            device_kind,
            torch,
            True,
        )
        assert metrics is not None
        repeats.append(metrics)

    summary = build_summary(args, model_dir, args.device, prompts, repeats, str(vllm.__version__))
    print_summary(summary, args.json)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vllm_generate_benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
