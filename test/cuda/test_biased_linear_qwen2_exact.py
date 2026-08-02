#!/usr/bin/env python3
"""Failure-first exact CUDA biased-linear fixtures at Qwen2 dimensions."""

import torch
import torch.nn.functional as F

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
    print("QWEN2_BIASED_LINEAR_EXACT_PASS")


if __name__ == "__main__":
    main()
