#!/usr/bin/env python3
import json

import torch

import llaisys
from test_ops_cuda import fetch, llaisys_tensor


SEED = 20260731
WIDTH = 896
EPSILON = 1e-6


def main():
    assert torch.__version__ == "2.4.0+cu124"
    assert torch.version.cuda == "12.4"
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    assert torch.cuda.current_device() == 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)

    cases = []
    total_mismatches = 0
    for rows, seed in (
        (1, 92),
        (2, 8),
        (4, SEED),
        (8, SEED + 1),
        (32, SEED + 2),
    ):
        generator = torch.Generator().manual_seed(seed)
        source = torch.randn((rows, WIDTH), generator=generator).to(
            torch.bfloat16
        )
        weight = (
            torch.randn((WIDTH,), generator=generator).to(torch.bfloat16)
            * 0.03125
            + 1
        )
        source_cuda = source.cuda()
        weight_cuda = weight.cuda()
        normalized = source_cuda.float() * torch.rsqrt(
            source_cuda.float().pow(2).mean(-1, keepdim=True) + EPSILON
        )
        expected = weight_cuda * normalized.to(torch.bfloat16)

        source_llaisys = llaisys_tensor(source, llaisys.DeviceType.NVIDIA)
        weight_llaisys = llaisys_tensor(weight, llaisys.DeviceType.NVIDIA)
        output_llaisys = llaisys.Tensor(
            (rows, WIDTH),
            dtype=llaisys.DataType.BF16,
            device=llaisys.DeviceType.NVIDIA,
            device_id=0,
        )
        llaisys.Ops.rms_norm(
            output_llaisys, source_llaisys, weight_llaisys, EPSILON
        )
        actual = fetch(output_llaisys)
        expected_cpu = expected.cpu()
        delta = (actual.float() - expected_cpu.float()).abs()
        mismatches = int((actual != expected_cpu).sum())
        total_mismatches += mismatches
        cases.append(
            {
                "rows": rows,
                "seed": seed,
                "mismatches": mismatches,
                "max_abs": float(delta.max()),
            }
        )

    print(
        "RMS_NORM_QWEN2_WIDTH="
        + json.dumps(
            {
                "dtype": "bf16",
                "epsilon": EPSILON,
                "width": WIDTH,
                "cases": cases,
                "total_mismatches": total_mismatches,
            },
            sort_keys=True,
        )
    )
    assert total_mismatches == 0, cases
    print("M6A CUDA RMSNORM QWEN2 WIDTH PASS")


if __name__ == "__main__":
    main()
