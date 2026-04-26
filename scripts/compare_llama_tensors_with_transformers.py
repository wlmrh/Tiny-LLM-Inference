#!/usr/bin/env python3
import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


SKIP_CODE = 77


def import_deps():
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        print(f"missing Python dependency for tensor alignment: {exc}")
        return None, None, None
    return torch, AutoConfig, AutoModelForCausalLM


def read_cpp_tensor(path: Path, torch):
    data = path.read_bytes()
    if len(data) < 4:
        raise RuntimeError(f"invalid tensor dump: {path}")
    rank = struct.unpack_from("<i", data, 0)[0]
    if rank < 0 or rank > 8:
        raise RuntimeError(f"invalid tensor rank {rank}: {path}")
    offset = 4
    shape = []
    for _ in range(rank):
        shape.append(struct.unpack_from("<q", data, offset)[0])
        offset += 8
    numel = 1
    for dim in shape:
        numel *= dim
    expected_size = offset + numel * 4
    if len(data) != expected_size:
        raise RuntimeError(f"invalid tensor byte size for {path}: got {len(data)}, expected {expected_size}")
    return torch.frombuffer(bytearray(data[offset:]), dtype=torch.float32).clone().reshape(shape)


def run_cpp_dump(binary: Path, model_dir: Path, output_dir: Path, tokens):
    run = subprocess.run(
        [str(binary), str(model_dir), str(output_dir), *[str(token) for token in tokens]],
        check=False,
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        sys.stderr.write(run.stdout)
        sys.stderr.write(run.stderr)
        raise RuntimeError(f"llama_tensor_dump failed with exit code {run.returncode}")


def layer_name(layer_id: int, suffix: str) -> str:
    return f"layer_{layer_id:02d}_{suffix}"


def squeeze_token_batch(tensor):
    tensor = tensor.detach().to(dtype=tensor.float().dtype, device="cpu").float()
    if tensor.dim() >= 3 and tensor.shape[0] == 1:
        return tensor.squeeze(0).contiguous()
    return tensor.contiguous()


def collect_transformers_tensors(model, input_ids, position_ids, torch):
    captures = {}
    handles = []
    pending_qkv = {}

    def save(name):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            captures[name] = squeeze_token_batch(output)
        return hook

    def save_pre(name):
        def hook(_module, inputs):
            captures[name] = squeeze_token_batch(inputs[0])
        return hook

    def save_qkv(layer_id, kind):
        def hook(_module, _inputs, output):
            layer_values = pending_qkv.setdefault(layer_id, {})
            layer_values[kind] = squeeze_token_batch(output)
            if {"q", "k", "v"}.issubset(layer_values):
                captures[layer_name(layer_id, "qkv")] = torch.cat(
                    [layer_values["q"], layer_values["k"], layer_values["v"]],
                    dim=-1,
                ).contiguous()
        return hook

    base = model.model
    handles.append(base.embed_tokens.register_forward_hook(save("00_embed")))
    for layer_id, layer in enumerate(base.layers):
        handles.append(layer.input_layernorm.register_forward_hook(save(layer_name(layer_id, "input_norm"))))
        handles.append(layer.self_attn.q_proj.register_forward_hook(save_qkv(layer_id, "q")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(save_qkv(layer_id, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(save_qkv(layer_id, "v")))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(save_pre(layer_name(layer_id, "attn_output"))))
        handles.append(layer.self_attn.o_proj.register_forward_hook(save(layer_name(layer_id, "attn_proj"))))
        handles.append(layer.post_attention_layernorm.register_forward_hook(save(layer_name(layer_id, "post_attn_norm"))))
        handles.append(layer.mlp.gate_proj.register_forward_hook(save(layer_name(layer_id, "mlp_gate"))))
        handles.append(layer.mlp.up_proj.register_forward_hook(save(layer_name(layer_id, "mlp_up"))))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(save_pre(layer_name(layer_id, "mlp_activated"))))
        handles.append(layer.mlp.down_proj.register_forward_hook(save(layer_name(layer_id, "mlp_down"))))
        handles.append(layer.register_forward_hook(save(layer_name(layer_id, "output"))))
    handles.append(base.norm.register_forward_hook(save("final_norm")))
    handles.append(model.lm_head.register_forward_hook(save("logits")))

    with torch.no_grad():
        model(input_ids=input_ids, position_ids=position_ids, use_cache=False)

    for handle in handles:
        handle.remove()
    return captures


def compare_tensor(name, actual, expected, atol, rtol, torch):
    if tuple(actual.shape) != tuple(expected.shape):
        return f"{name}: shape mismatch cpp={tuple(actual.shape)} transformers={tuple(expected.shape)}"
    if torch.allclose(actual, expected, atol=atol, rtol=rtol):
        return None
    diff = (actual - expected).abs()
    tolerance = atol + rtol * expected.abs()
    excess = diff - tolerance
    flat_index = int(excess.reshape(-1).argmax().item())
    max_excess = float(excess.reshape(-1)[flat_index].item())
    max_diff = float(diff.reshape(-1)[flat_index].item())
    allowed = float(tolerance.reshape(-1)[flat_index].item())
    cpp_value = float(actual.reshape(-1)[flat_index].item())
    hf_value = float(expected.reshape(-1)[flat_index].item())
    return (
        f"{name}: max_tolerance_excess={max_excess:.8g} "
        f"abs_diff={max_diff:.8g} allowed={allowed:.8g} flat_index={flat_index} "
        f"cpp={cpp_value:.8g} transformers={hf_value:.8g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Tiny-LLM intermediate LLaMA tensors with Transformers")
    parser.add_argument("--dump-binary", required=True, help="Path to compiled llama_tensor_dump")
    parser.add_argument("--model-dir", default=os.environ.get("TINYLLM_HF_TINY_LLAMA_DIR", ""))
    parser.add_argument("--tokens", nargs="+", type=int, default=None)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    torch, auto_config, auto_model = import_deps()
    if torch is None:
        return SKIP_CODE

    if not args.model_dir:
        print("TINYLLM_HF_TINY_LLAMA_DIR is not set; skipping tensor alignment comparison.")
        return SKIP_CODE

    model_dir = Path(args.model_dir).expanduser()
    if not model_dir.exists():
        print(f"model directory does not exist: {model_dir}; skipping tensor alignment comparison.")
        return SKIP_CODE
    dump_binary = Path(args.dump_binary)
    if not dump_binary.exists():
        raise RuntimeError(f"dump binary does not exist: {dump_binary}")

    config = auto_config.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    tokens = args.tokens
    if tokens is None:
        bos_id = config.bos_token_id if isinstance(config.bos_token_id, int) else 0
        tokens = [bos_id, 7 if config.vocab_size > 7 else min(config.vocab_size - 1, bos_id + 1)]

    input_ids = torch.tensor([tokens], dtype=torch.long)
    position_ids = torch.arange(len(tokens), dtype=torch.long).unsqueeze(0)
    model = auto_model.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()

    with tempfile.TemporaryDirectory(prefix="tinyllm_tensor_alignment_") as tmp_dir:
        output_dir = Path(tmp_dir)
        run_cpp_dump(dump_binary, model_dir, output_dir, tokens)
        cpp_tensors = {
            path.stem: read_cpp_tensor(path, torch)
            for path in sorted(output_dir.glob("*.bin"))
        }

    hf_tensors = collect_transformers_tensors(model, input_ids, position_ids, torch)
    layer_suffixes = [
        "input_norm",
        "qkv",
        "attn_output",
        "attn_proj",
        "post_attn_norm",
        "mlp_gate",
        "mlp_up",
        "mlp_activated",
        "mlp_down",
        "output",
    ]
    expected_order = ["00_embed"]
    for layer_id in range(int(config.num_hidden_layers)):
        expected_order.extend(layer_name(layer_id, suffix) for suffix in layer_suffixes)
    expected_order.extend(["final_norm", "logits"])
    ordered_names = [name for name in expected_order if name in cpp_tensors and name in hf_tensors]
    failures = []
    for name in ordered_names:
        failure = compare_tensor(name, cpp_tensors[name], hf_tensors[name], args.atol, args.rtol, torch)
        if failure is not None:
            failures.append(failure)
            break

    if failures:
        print("first tensor alignment mismatch:")
        print(failures[0])
        return 1

    missing = sorted(set(cpp_tensors.keys()) - set(hf_tensors.keys()))
    print(f"Compared {len(ordered_names)} tensors for tokens {tokens}; all matched.")
    if missing:
        print(f"Skipped tensors without Transformers hook: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"compare_llama_tensors_with_transformers failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
