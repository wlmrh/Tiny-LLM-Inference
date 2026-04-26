#!/usr/bin/env python3
import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple


SKIP_CODE = 77


def import_transformers_deps():
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        print(f"missing Python dependency for transformers logits comparison: {exc}")
        return None, None, None
    return torch, AutoConfig, AutoModelForCausalLM


def select_token_ids(config) -> Tuple[int, int]:
    vocab_size = int(config.vocab_size)
    if vocab_size <= 0:
        raise RuntimeError("config.vocab_size must be positive")

    bos_id = getattr(config, "bos_token_id", None)
    if isinstance(bos_id, int) and 0 <= bos_id < vocab_size:
        first = bos_id
    else:
        first = 0

    second = 7 if vocab_size > 7 else min(vocab_size - 1, first + 1)
    return first, second


def read_cpp_logits(path: Path, torch):
    data = path.read_bytes()
    if len(data) < 8:
        raise RuntimeError(f"invalid C++ logits file, expected header: {path}")

    batch_size, vocab_size = struct.unpack_from("<ii", data, 0)
    if batch_size <= 0 or vocab_size <= 0:
        raise RuntimeError(f"invalid logits shape from C++: [{batch_size}, {vocab_size}]")

    expected_bytes = 8 + batch_size * vocab_size * 4
    if len(data) != expected_bytes:
        raise RuntimeError(
            f"invalid C++ logits file size: got {len(data)} bytes, expected {expected_bytes}"
        )

    logits = torch.frombuffer(bytearray(data[8:]), dtype=torch.float32).clone()
    return logits.reshape(batch_size, vocab_size)


def run_cpp_dump(binary: Path, model_dir: Path, output_path: Path, token0: int, token1: int) -> None:
    run = subprocess.run(
        [str(binary), str(model_dir), str(output_path), str(token0), str(token1)],
        check=False,
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        sys.stderr.write(run.stdout)
        sys.stderr.write(run.stderr)
        raise RuntimeError(f"llama_logits_dump failed with exit code {run.returncode}")


def compare_logits(actual, expected, atol: float, rtol: float, torch) -> None:
    if tuple(actual.shape) != tuple(expected.shape):
        raise RuntimeError(f"logits shape mismatch: cpp={tuple(actual.shape)} transformers={tuple(expected.shape)}")

    actual_top = actual.argmax(dim=1)
    expected_top = expected.argmax(dim=1)
    values_match = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    top_tokens_match = torch.equal(actual_top, expected_top)
    if values_match and top_tokens_match:
        return

    diff = (actual - expected).abs()
    flat_index = int(diff.reshape(-1).argmax().item())
    row = flat_index // actual.shape[1]
    col = flat_index % actual.shape[1]
    max_diff = float(diff[row, col].item())

    print(
        "logits mismatch: "
        f"max_abs_diff={max_diff} row={row} col={col} "
        f"cpp={float(actual[row, col].item())} "
        f"transformers={float(expected[row, col].item())}"
    )
    for idx in range(actual.shape[0]):
        print(
            f"row {idx}: "
            f"cpp_top={int(actual_top[idx].item())} "
            f"transformers_top={int(expected_top[idx].item())}"
        )
    raise RuntimeError("C++ logits do not match Transformers logits")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Tiny-LLM LLaMA logits with Transformers")
    parser.add_argument("--dump-binary", required=True, help="Path to the compiled llama_logits_dump helper")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    model_dir_text = os.environ.get("TINYLLM_HF_TINY_LLAMA_DIR", "")
    if not model_dir_text:
        print("TINYLLM_HF_TINY_LLAMA_DIR is not set; skipping Transformers logits comparison.")
        return SKIP_CODE

    torch, auto_config, auto_model = import_transformers_deps()
    if torch is None:
        return SKIP_CODE

    model_dir = Path(model_dir_text)
    if not model_dir.exists():
        raise RuntimeError(f"model directory does not exist: {model_dir}")

    dump_binary = Path(args.dump_binary)
    if not dump_binary.exists():
        raise RuntimeError(f"dump binary does not exist: {dump_binary}")

    config = auto_config.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    token0, token1 = select_token_ids(config)

    with tempfile.TemporaryDirectory(prefix="tinyllm_logits_") as tmp_dir:
        cpp_logits_path = Path(tmp_dir) / "cpp_logits.bin"
        run_cpp_dump(dump_binary, model_dir, cpp_logits_path, token0, token1)
        cpp_logits = read_cpp_logits(cpp_logits_path, torch)

    model = auto_model.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()

    input_ids = torch.tensor([[token0, token1]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1]], dtype=torch.long)
    with torch.no_grad():
        expected_logits = model(input_ids=input_ids, position_ids=position_ids).logits[0].to(torch.float32).cpu()

    compare_logits(cpp_logits, expected_logits, args.atol, args.rtol, torch)
    print(
        "Tiny-LLM C++ logits match Transformers for "
        f"tokens [{token0}, {token1}] with shape {tuple(cpp_logits.shape)}."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"compare_llama_logits_with_transformers failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
