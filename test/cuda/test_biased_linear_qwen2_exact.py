#!/usr/bin/env python3
"""Failure-first exact CUDA biased-linear fixtures at Qwen2 dimensions."""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

import llaisys
from test_ops_cuda import empty, fetch, paired_inputs


def run_case(output_features, seed):
    generator = torch.Generator().manual_seed(seed)
    batch, input_features = 8, 896
    source = torch.randn(
        (batch, input_features), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    weight = torch.randn(
        (output_features, input_features), generator=generator,
        dtype=torch.float32,
    ).div(32).to(torch.bfloat16)
    bias = torch.randn(
        (output_features,), generator=generator, dtype=torch.float32
    ).div(16).to(torch.bfloat16)
    tensors = paired_inputs([source, weight, bias])["cuda"]
    output = empty(
        (batch, output_features), "bf16", llaisys.DeviceType.NVIDIA
    )
    llaisys.Ops.linear(output, *tensors)
    actual = fetch(output)
    expected = F.linear(source.cuda(), weight.cuda(), bias.cuda()).cpu()
    mismatch = actual != expected
    delta = (actual.float() - expected.float()).abs()
    count = int(mismatch.sum())
    maximum = float(delta.max())
    print(
        f"output_features={output_features} mismatches={count} max_abs={maximum}"
    )
    assert count == 0


def main():
    assert torch.__version__ == "2.4.0+cu124"
    assert torch.cuda.device_count() == 1
    torch.use_deterministic_algorithms(True)
    run_case(128, 20260731)
    run_case(896, 20260732)
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
    tokens = list(tokenizer.apply_chat_template(
        [message], tokenize=True, add_generation_prompt=True
    ))[:8]
    os.environ["LLAISYS_QWEN2_TRACE_LAYER"] = "1"
    model = llaisys.Qwen2(str(model_dir), device="cuda", dtype="bf16")
    source = model.forward_trace(tokens)["diagnostic_attn_norm"]
    weights = load_file(model_dir / "model.safetensors")
    prefix = "model.layers.1.self_attn.v_proj"
    weight = weights[f"{prefix}.weight"]
    bias = weights[f"{prefix}.bias"]
    tensors = paired_inputs([source, weight, bias])["cuda"]
    output = empty(
        (source.shape[0], bias.shape[0]), "bf16", llaisys.DeviceType.NVIDIA
    )
    llaisys.Ops.linear(output, *tensors)
    actual = fetch(output)
    expected = F.linear(source.cuda(), weight.cuda(), bias.cuda()).cpu()
    mismatch = actual != expected
    delta = (actual.float() - expected.float()).abs()
    count = int(mismatch.sum())
    maximum = float(delta.max())
    print(f"real_layer1_v mismatches={count} max_abs={maximum}")
    assert count == 0
    print("QWEN2_BIASED_LINEAR_EXACT_PASS")


if __name__ == "__main__":
    main()
