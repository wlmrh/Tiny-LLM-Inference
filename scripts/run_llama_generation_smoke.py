#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SKIP_CODE = 77


EXPECTED_IDS = {
    "hello": [28, 476, 57, 5248, 22657, 28, 339, 3683],
    "tiny llm inference": [30, 198, 198, 504, 450, 34519, 23630, 314],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Tiny-LLM greedy generation smoke test")
    parser.add_argument("--engine-binary", required=True)
    parser.add_argument("--model-dir", default=os.environ.get("TINYLLM_HF_TINY_LLAMA_DIR", ""))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", action="append", dest="prompts", default=[])
    parser.add_argument("--check-known-ids", action="store_true")
    args = parser.parse_args()

    if not args.model_dir:
        print("TINYLLM_HF_TINY_LLAMA_DIR is not set; skipping generation smoke.")
        return SKIP_CODE

    model_dir = Path(args.model_dir).expanduser()
    if not model_dir.exists():
        print(f"model directory does not exist: {model_dir}; skipping generation smoke.")
        return SKIP_CODE

    engine_binary = Path(args.engine_binary)
    if not engine_binary.exists():
        raise RuntimeError(f"engine binary does not exist: {engine_binary}")

    prompts = args.prompts or ["hello", "tiny llm inference"]
    command = [
        str(engine_binary),
        "--device",
        args.device,
        str(model_dir),
        str(args.max_new_tokens),
        *prompts,
    ]
    run = subprocess.run(command, check=False, capture_output=True, text=True)
    if run.returncode != 0:
        sys.stderr.write(run.stdout)
        sys.stderr.write(run.stderr)
        raise RuntimeError(f"llama_engine_generate failed with exit code {run.returncode}")

    results = []
    for line in run.stdout.splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))

    if len(results) != len(prompts):
        raise RuntimeError(f"expected {len(prompts)} generation results, got {len(results)}")

    for prompt, result in zip(prompts, results):
        generated_ids = result.get("generated_token_ids")
        if not isinstance(generated_ids, list) or len(generated_ids) != args.max_new_tokens:
            raise RuntimeError(f"unexpected generated_token_ids for prompt {prompt!r}: {generated_ids}")
        if result.get("finish_reason") != "length":
            raise RuntimeError(f"unexpected finish_reason for prompt {prompt!r}: {result.get('finish_reason')!r}")
        output = result.get("output", "")
        if not isinstance(output, str) or not output.startswith(prompt):
            raise RuntimeError(f"unexpected output for prompt {prompt!r}: {output!r}")
        if args.check_known_ids:
            expected = EXPECTED_IDS.get(prompt)
            if expected is not None and generated_ids != expected:
                raise RuntimeError(
                    f"generated ids mismatch for prompt {prompt!r}: got {generated_ids}, expected {expected}"
                )
        print(f"[OK] {args.device} prompt={prompt!r} ids={generated_ids} output={output!r}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"run_llama_generation_smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
