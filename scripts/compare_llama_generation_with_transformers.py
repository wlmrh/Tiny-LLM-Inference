#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SKIP_CODE = 77


def import_deps():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"missing Python dependency for generation comparison: {exc}")
        return None, None, None
    return torch, AutoModelForCausalLM, AutoTokenizer


def run_cpp_generate(binary: Path, model_dir: Path, max_new_tokens: int, prompts):
    run = subprocess.run(
        [str(binary), str(model_dir), str(max_new_tokens), *prompts],
        check=False,
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        sys.stderr.write(run.stdout)
        sys.stderr.write(run.stderr)
        raise RuntimeError(f"llama_engine_generate failed with exit code {run.returncode}")

    results = []
    for line in run.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        results.append(json.loads(line))
    return results


def run_transformers_generate(model_dir: Path, max_new_tokens: int, prompts, torch, auto_model, auto_tokenizer):
    tokenizer = auto_tokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = auto_model.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()

    results = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[0]

        generated_ids = output_ids[input_ids.shape[1]:].tolist()
        results.append({
            "prompt": prompt,
            "output": tokenizer.decode(output_ids, skip_special_tokens=False),
            "generated_token_ids": generated_ids,
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Tiny-LLM engine greedy generation with Transformers")
    parser.add_argument("--engine-binary", required=True, help="Path to compiled llama_engine_generate")
    parser.add_argument("--model-dir", default=os.environ.get("TINYLLM_HF_TINY_LLAMA_DIR", ""))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", action="append", dest="prompts", default=[])
    parser.add_argument("--show-only", action="store_true", help="Print both outputs without failing on mismatch")
    args = parser.parse_args()

    torch, auto_model, auto_tokenizer = import_deps()
    if torch is None:
        return SKIP_CODE

    if not args.model_dir:
        print("TINYLLM_HF_TINY_LLAMA_DIR is not set; skipping generation comparison.")
        return SKIP_CODE

    model_dir = Path(args.model_dir).expanduser()
    if not model_dir.exists():
        print(f"model directory does not exist: {model_dir}; skipping generation comparison.")
        return SKIP_CODE

    engine_binary = Path(args.engine_binary)
    if not engine_binary.exists():
        raise RuntimeError(f"engine binary does not exist: {engine_binary}")

    prompts = args.prompts or ["hello", "tiny llm inference"]
    cpp_results = run_cpp_generate(engine_binary, model_dir, args.max_new_tokens, prompts)
    hf_results = run_transformers_generate(
        model_dir,
        args.max_new_tokens,
        prompts,
        torch,
        auto_model,
        auto_tokenizer,
    )

    failed = False
    for cpp, hf in zip(cpp_results, hf_results):
        token_match = cpp["generated_token_ids"] == hf["generated_token_ids"]
        text_match = cpp["output"] == hf["output"]
        status = "MATCH" if token_match and text_match else "MISMATCH"
        print(f"[{status}] prompt: {cpp['prompt']!r}")
        print(f"  tiny-llm ids : {cpp['generated_token_ids']}")
        print(f"  pytorch ids  : {hf['generated_token_ids']}")
        print(f"  tiny-llm text: {cpp['output']!r}")
        print(f"  pytorch text : {hf['output']!r}")
        if not token_match or not text_match:
            failed = True

    if failed and not args.show_only:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"compare_llama_generation_with_transformers failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
