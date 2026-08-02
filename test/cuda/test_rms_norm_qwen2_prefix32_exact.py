#!/usr/bin/env python3
"""Exact CUDA RMSNorm fixture from Qwen2 prefix-32 layer 22."""

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
    message = {
        "role": "user",
        "content": (
            "Count from one to ten in English, separated by commas, and then "
            "explain in one sentence why ten follows nine."
        ),
    }
    tokens = list(
        tokenizer.apply_chat_template(
            [message], tokenize=True, add_generation_prompt=True
        )
    )[:32]
    layer = 22
    os.environ["LLAISYS_QWEN2_TRACE_LAYER"] = str(layer)
    model = llaisys.Qwen2(str(model_dir), device="cuda", dtype="bf16")
    source = model.forward_trace(tokens)["diagnostic_post_attention"]
    weight = load_file(model_dir / "model.safetensors")[
        f"model.layers.{layer}.post_attention_layernorm.weight"
    ]
    source_cuda = source.cuda()
    weight_cuda = weight.cuda()
    normalized = source_cuda.float() * torch.rsqrt(
        source_cuda.float().pow(2).mean(-1, keepdim=True) + 1e-6
    )
    expected = (normalized.to(torch.bfloat16) * weight_cuda).cpu()
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
    print(
        f"prefix32_layer22_mlp_rms mismatches={mismatches} max_abs={maximum}"
    )
    assert mismatches == 0
    print("QWEN2_PREFIX32_RMS_NORM_EXACT_PASS")


if __name__ == "__main__":
    main()
