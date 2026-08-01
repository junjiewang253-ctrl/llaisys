#!/usr/bin/env python3
"""Failure-first exact RoPE fixture for Qwen2-0.5B CUDA dimensions."""

import torch

import llaisys


def llaisys_tensor(value):
    result = llaisys.Tensor(
        tuple(value.shape), dtype=llaisys.DataType.BF16,
        device=llaisys.DeviceType.NVIDIA, device_id=0,
    )
    result.load(value.contiguous().data_ptr())
    return result


def fetch(value):
    result = torch.empty(tuple(value.shape()), dtype=torch.bfloat16)
    llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA).memcpy_sync(
        result.data_ptr(), value.data_ptr(), result.numel() * result.element_size(),
        llaisys.MemcpyKind.D2H,
    )
    return result


def main():
    assert torch.__version__ == "2.4.0+cu124"
    assert torch.cuda.device_count() == 1
    torch.manual_seed(20260731)
    sequence, heads, head_dim = 8, 14, 64
    source = (
        torch.randn((sequence, heads, head_dim), dtype=torch.float32)
        * 8.0
    ).to(torch.bfloat16)
    positions = torch.arange(sequence, dtype=torch.int64)

    inv_freq = 1.0 / (
        10000.0 ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim
        )
    )
    frequencies = torch.outer(positions.float(), inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine = embedding.cos().to(torch.bfloat16).cuda().unsqueeze(1)
    sine = embedding.sin().to(torch.bfloat16).cuda().unsqueeze(1)
    source_cuda = source.cuda()
    rotated = torch.cat(
        (-source_cuda[..., head_dim // 2 :], source_cuda[..., : head_dim // 2]),
        dim=-1,
    )
    expected = source_cuda * cosine + rotated * sine

    input_tensor = llaisys_tensor(source)
    position_tensor = llaisys.Tensor(
        tuple(positions.shape), dtype=llaisys.DataType.I64,
        device=llaisys.DeviceType.NVIDIA, device_id=0,
    )
    position_tensor.load(positions.contiguous().data_ptr())
    output = llaisys.Tensor(
        tuple(source.shape), dtype=llaisys.DataType.BF16,
        device=llaisys.DeviceType.NVIDIA, device_id=0,
    )
    llaisys.Ops.rope(output, input_tensor, position_tensor, 10000.0)
    actual = fetch(output)
    expected_cpu = expected.cpu()
    mismatch = actual != expected_cpu
    delta = (actual.float() - expected_cpu.float()).abs()
    print(f"mismatches={int(mismatch.sum())}")
    print(f"max_abs={float(delta.max())}")
    assert torch.equal(actual, expected_cpu)
    print("QWEN2_ROPE_EXACT_PASS")


if __name__ == "__main__":
    main()
