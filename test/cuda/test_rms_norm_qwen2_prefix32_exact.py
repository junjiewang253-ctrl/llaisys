#!/usr/bin/env python3
"""Exact CUDA RMSNorm fixture from a frozen Qwen2 trace boundary."""

import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

import llaisys
from test_ops_cuda import fetch, llaisys_tensor


def main():
    assert torch.__version__ == "2.4.0+cu124"
    assert torch.cuda.device_count() == 1
    torch.manual_seed(20260731)
    torch.cuda.manual_seed_all(20260731)
    torch.use_deterministic_algorithms(True)
    model_dir = Path(os.environ["LLAISYS_M6B_MODEL_DIR"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    prompts = {
        "P1": {"role": "user", "content": "你好，请用一句话介绍你自己。"},
        "P2": {
            "role": "user",
            "content": "What is 17 plus 25? Answer with only the number.",
        },
        "P3": {
            "role": "user",
            "content": (
                "Count from one to ten in English, separated by commas, and then "
                "explain in one sentence why ten follows nine."
            ),
        },
    }
    prompt_id = os.environ.get("LLAISYS_M6B_RMS_PROMPT", "P3")
    message = prompts[prompt_id]
    prefix_text = os.environ.get(
        "LLAISYS_M6B_RMS_PREFIX", "32" if prompt_id == "P3" else ""
    )
    tokens = list(
        tokenizer.apply_chat_template(
            [message], tokenize=True, add_generation_prompt=True
        )
    )
    if prefix_text:
        tokens = tokens[: int(prefix_text)]
    append_text = os.environ.get("LLAISYS_M6B_RMS_APPEND_TOKENS", "")
    if append_text:
        tokens.extend(int(token) for token in append_text.split(","))
    layer = int(os.environ.get("LLAISYS_M6B_RMS_LAYER", "22"))
    boundary = os.environ.get("LLAISYS_M6B_RMS_BOUNDARY", "mlp")
    assert boundary in {"attn", "mlp"}
    os.environ["LLAISYS_QWEN2_TRACE_LAYER"] = str(layer)
    model = llaisys.Qwen2(str(model_dir), device="cuda", dtype="bf16")
    trace = model.forward_trace(tokens)
    weights = load_file(model_dir / "model.safetensors")
    if boundary == "attn":
        source = trace["embedding"] if layer == 0 else trace[f"layer.{layer - 1}.output"]
        weight = weights[f"model.layers.{layer}.input_layernorm.weight"]
    else:
        source = trace["diagnostic_post_attention"]
        weight = weights[f"model.layers.{layer}.post_attention_layernorm.weight"]
    source_cuda = source.cuda()
    weight_cuda = weight.cuda()
    source_float = source_cuda.float()
    normalized = source_float * torch.rsqrt(
        source_float.pow(2).mean(-1, keepdim=True) + 1e-6
    )
    expected = (normalized.to(torch.bfloat16) * weight_cuda).cpu()
    normalized_mul = source_float * torch.rsqrt(
        (source_float * source_float).mean(-1, keepdim=True) + 1e-6
    )
    expected_mul = (normalized_mul.to(torch.bfloat16) * weight_cuda).cpu()
    source_ll = llaisys_tensor(source, llaisys.DeviceType.NVIDIA)
    weight_ll = llaisys_tensor(weight, llaisys.DeviceType.NVIDIA)
    output_ll = llaisys.Tensor(
        source.shape,
        dtype=llaisys.DataType.BF16,
        device=llaisys.DeviceType.NVIDIA,
        device_id=0,
    )
    llaisys.Ops.rms_norm(output_ll, source_ll, weight_ll, 1e-6)
    actual = fetch(output_ll)
    delta = (actual.float() - expected.float()).abs()
    mismatches = int((actual != expected).sum())
    maximum = float(delta.max())
    pow_mul_mismatches = int((expected != expected_mul).sum())
    actual_mul_mismatches = int((actual != expected_mul).sum())
    print(
        f"{prompt_id}_length{len(tokens)}_layer{layer}_{boundary}_rms "
        f"mismatches={mismatches} max_abs={maximum}"
    )
    print(
        f"pow_vs_mul_mismatches={pow_mul_mismatches} "
        f"actual_vs_mul_mismatches={actual_mul_mismatches}"
    )
    assert mismatches == 0
    print("QWEN2_RMS_NORM_EXACT_PASS")


if __name__ == "__main__":
    main()
