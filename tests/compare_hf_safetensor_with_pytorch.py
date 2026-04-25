#!/usr/bin/env python3
import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List

import torch
from safetensors.torch import load_file


@dataclass
class TensorDigest:
    key: str
    shape: List[int]
    numel: int
    sum_value: float
    l2_value: float
    first_values: List[float]


def parse_shape(shape_text: str) -> List[int]:
    if shape_text == "scalar":
        return []
    if not shape_text:
        return []
    return [int(part) for part in shape_text.split(",") if part]


def parse_first_values(first_text: str) -> List[float]:
    if not first_text:
        return []
    return [float(part) for part in first_text.split(",") if part]


def parse_dump_output(stdout_text: str) -> Dict[str, TensorDigest]:
    digests: Dict[str, TensorDigest] = {}
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) != 6:
            raise RuntimeError(f"invalid dump line format: {line}")

        key, shape_text, numel_text, sum_text, l2_text, first_text = parts
        digest = TensorDigest(
            key=key,
            shape=parse_shape(shape_text),
            numel=int(numel_text),
            sum_value=float(sum_text),
            l2_value=float(l2_text),
            first_values=parse_first_values(first_text),
        )
        digests[key] = digest

    if not digests:
        raise RuntimeError("no tensor digests were parsed from C++ dump output")

    return digests


def close_enough(lhs: float, rhs: float, atol: float, rtol: float) -> bool:
    return math.isclose(lhs, rhs, abs_tol=atol, rel_tol=rtol)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Tiny-LLM safetensor loader output with PyTorch")
    parser.add_argument(
        "--model-dir",
        default="/Users/tangqi/weights",
        help="HuggingFace model directory that contains model.safetensors",
    )
    parser.add_argument(
        "--weight-file",
        default="model.safetensors",
        help="SafeTensor filename inside model dir",
    )
    parser.add_argument(
        "--dump-binary",
        required=True,
        help="Path to compiled hf_safetensor_dump executable",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()

    run = subprocess.run(
        [args.dump_binary, args.model_dir, args.weight_file],
        check=False,
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        sys.stderr.write(run.stderr)
        raise RuntimeError("hf_safetensor_dump execution failed")

    cpp_digests = parse_dump_output(run.stdout)

    safetensor_path = f"{args.model_dir}/{args.weight_file}"
    pytorch_weights = load_file(safetensor_path, device="cpu")

    failures: List[str] = []
    for key, digest in cpp_digests.items():
        if key not in pytorch_weights:
            failures.append(f"missing key in pytorch safetensors: {key}")
            continue

        tensor = pytorch_weights[key].detach().cpu().to(torch.float32).contiguous()
        shape = list(tensor.shape)
        if shape != digest.shape:
            failures.append(f"shape mismatch for {key}: cpp={digest.shape} pytorch={shape}")
            continue

        flat = tensor.view(-1).to(torch.float64)
        numel = int(flat.numel())
        if numel != digest.numel:
            failures.append(f"numel mismatch for {key}: cpp={digest.numel} pytorch={numel}")
            continue

        sum_value = float(flat.sum().item())
        l2_value = float((flat * flat).sum().item())

        if not close_enough(sum_value, digest.sum_value, args.atol, args.rtol):
            failures.append(
                f"sum mismatch for {key}: cpp={digest.sum_value} pytorch={sum_value}"
            )

        if not close_enough(l2_value, digest.l2_value, args.atol, args.rtol):
            failures.append(
                f"l2 mismatch for {key}: cpp={digest.l2_value} pytorch={l2_value}"
            )

        expected_first = flat[: len(digest.first_values)].tolist()
        for idx, (lhs, rhs) in enumerate(zip(digest.first_values, expected_first)):
            if not close_enough(lhs, rhs, args.atol, args.rtol):
                failures.append(
                    f"first value mismatch for {key}[{idx}]: cpp={lhs} pytorch={rhs}"
                )

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print(f"Compared {len(cpp_digests)} tensors. Tiny-LLM safetensor loader matches PyTorch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
