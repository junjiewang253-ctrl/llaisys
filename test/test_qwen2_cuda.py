#!/usr/bin/env python3
"""Failure-first and regression entry point for the M6-B Qwen2 CUDA gate."""

import os
from pathlib import Path

import llaisys


def main():
    model_dir = Path(os.environ["LLAISYS_M6B_TINY_DIR"])
    model = llaisys.Qwen2(str(model_dir), device="cuda", dtype="f32")

    # A wrapper that silently constructs a CPU model is a production fallback,
    # not a CUDA implementation.  This assertion is deliberately before any
    # numerical gate so the original code fails for the intended reason.
    assert model.device == llaisys.DeviceType.NVIDIA, model.device
    assert model.device_id == 0

    result = model.forward_trace([1, 5, 7, 9])
    assert result["logits"].shape == (4, 32)
    assert isinstance(result["greedy_token"], int)

    model.reset_cache()
    cached = model.forward_cached_trace([1, 5, 7, 9])
    assert cached["logits"].shape == (4, 32)
    assert model.cache_info()["cursor"] == 4
    print("M6B QWEN2 CUDA FAILURE-FIRST CONTRACT PASS")


if __name__ == "__main__":
    main()
