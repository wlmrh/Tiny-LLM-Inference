#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_PROMPTS = ["hello", "tiny llm inference"]


class FirstTokenTimer:
    def __init__(self, synchronize):
        self._synchronize = synchronize
        self.start_time = 0.0
        self.first_token_time: Optional[float] = None

    def mark_start(self) -> None:
        self._synchronize()
        self.start_time = time.perf_counter()
        self.first_token_time = None

    def __call__(self, input_ids, scores):
        if self.first_token_time is None:
            self._synchronize()
            self.first_token_time = time.perf_counter()
        return scores


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
    parser = argparse.ArgumentParser(description="Benchmark Hugging Face Transformers greedy generation")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:<device_id>")
    parser.add_argument("--warmup", type=non_negative_int, default=1)
    parser.add_argument("--repeat", type=positive_int, default=3)
    parser.add_argument("--max-new-tokens", type=positive_int, default=8)
    parser.add_argument("--prompt", action="append", dest="prompts", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("model_dir")
    return parser.parse_args()


def validate_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise RuntimeError(f"model_dir is not a directory: {model_dir}")
    if not (model_dir / "config.json").exists():
        raise RuntimeError(f"model_dir must contain config.json: {model_dir}")
    if not (model_dir / "tokenizer.json").exists() and not (model_dir / "tokenizer.model").exists():
        raise RuntimeError(f"model_dir must contain tokenizer.json or tokenizer.model: {model_dir}")
    if not (model_dir / "model.safetensors").exists() and not any(model_dir.glob("*.safetensors")):
        raise RuntimeError(f"model_dir must contain model.safetensors or safetensors shards: {model_dir}")


def import_deps():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation.logits_process import LogitsProcessorList
    except ImportError as exc:
        raise RuntimeError(f"missing Python dependency for Transformers benchmark: {exc}") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList


def parse_device(device_text: str, torch) -> str:
    if device_text == "cpu":
        return "cpu"
    if device_text == "cuda":
        device = "cuda:0"
    elif device_text.startswith("cuda:"):
        suffix = device_text.split(":", 1)[1]
        if not suffix.isdigit():
            raise RuntimeError("device must be cpu, cuda, or cuda:<device_id>")
        device = f"cuda:{int(suffix)}"
    else:
        raise RuntimeError("device must be cpu, cuda, or cuda:<device_id>")

    if not torch.cuda.is_available():
        raise RuntimeError("--device cuda requires torch.cuda.is_available()")
    device_id = int(device.split(":", 1)[1])
    if device_id >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device_id} is not available; device_count={torch.cuda.device_count()}")
    return device


def make_synchronizer(device: str, torch):
    def synchronize() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
    return synchronize


def count_prompt_tokens(tokenizer, prompts: List[str]) -> int:
    total = 0
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        total += int(encoded["input_ids"].shape[1])
    return total


def load_model(model_dir: Path, device: str, torch, auto_model, auto_tokenizer):
    load_start = time.perf_counter()
    tokenizer = auto_tokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = auto_model.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    model.to(device)
    model.eval()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    load_ms = (time.perf_counter() - load_start) * 1000.0
    return tokenizer, model, load_ms


def generate_once(
    tokenizer,
    model,
    prompts: List[str],
    prompt_tokens: int,
    load_ms: float,
    args: argparse.Namespace,
    device: str,
    torch,
    logits_processor_list,
    measure: bool,
) -> Optional[Dict[str, Any]]:
    synchronize = make_synchronizer(device, torch)
    timer = FirstTokenTimer(synchronize)

    encoded = tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    timer.mark_start()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor_list([timer]),
        )
    synchronize()
    generation_end = time.perf_counter()

    generated_tokens = 0
    input_width = int(input_ids.shape[1])
    for row in output_ids:
        generated_tokens += max(0, int(row.shape[0]) - input_width)

    first_token_ms = 0.0
    if timer.first_token_time is not None:
        first_token_ms = (timer.first_token_time - timer.start_time) * 1000.0

    total_ms = (generation_end - timer.start_time) * 1000.0

    if not measure:
        return None

    return {
        "load_ms": load_ms,
        "total_ms": total_ms,
        "first_token_ms": first_token_ms,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
    }


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_summary(args: argparse.Namespace, model_dir: Path, device_text: str, prompts: List[str], repeats: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    return {
        "benchmark": "transformers_generate_benchmark",
        "backend": "transformers",
        "model": str(model_dir),
        "device": device_text,
        "prompt_count": len(prompts),
        "prompt_tokens": prompt_tokens,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "max_new_tokens": args.max_new_tokens,
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
    }


def print_summary(summary: Dict[str, Any], emit_json: bool) -> None:
    print("transformers_generate_benchmark")
    print(f"  model: {summary['model']}")
    print(f"  device: {summary['device']}")
    print(
        f"  prompts: {summary['prompt_count']}, warmup: {summary['warmup']}, "
        f"repeat: {summary['repeat']}, max_new_tokens: {summary['max_new_tokens']}"
    )
    print("  latency:")
    print(f"    avg_load_init_ms: {summary['avg_load_init_ms']:.3f}")
    print(f"    avg_total_latency_ms: {summary['avg_total_latency_ms']:.3f}")
    print(f"    avg_first_token_latency_ms: {summary['avg_first_token_latency_ms']:.3f}")
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
    if emit_json:
        print("  json: see final machine-readable line below")
        print(json.dumps(summary, separators=(",", ":")))


def main() -> int:
    args = parse_args()
    prompts = args.prompts or DEFAULT_PROMPTS
    model_dir = Path(args.model_dir).expanduser()
    validate_model_dir(model_dir)
    torch, auto_model, auto_tokenizer, logits_processor_list = import_deps()
    device = parse_device(args.device, torch)

    tokenizer, model, load_ms = load_model(model_dir, device, torch, auto_model, auto_tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    prompt_tokens = count_prompt_tokens(tokenizer, prompts)

    for _ in range(args.warmup):
        generate_once(tokenizer, model, prompts, prompt_tokens, load_ms, args, device, torch, logits_processor_list, False)

    repeats = []
    for _ in range(args.repeat):
        metrics = generate_once(tokenizer, model, prompts, prompt_tokens, load_ms, args, device, torch, logits_processor_list, True)
        assert metrics is not None
        repeats.append(metrics)

    summary = build_summary(args, model_dir, args.device, prompts, repeats)
    print_summary(summary, args.json)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"transformers_generate_benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
