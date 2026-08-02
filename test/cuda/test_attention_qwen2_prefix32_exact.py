#!/usr/bin/env python3
"""Failure-first exact CUDA attention fixture for Qwen2 prefix length 32."""

import math
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

import llaisys
from test_ops_cuda import empty, fetch, paired_inputs


def torch_attention(query, key, value):
    query = query.transpose(0, 1).unsqueeze(0).cuda()
    key = key.transpose(0, 1).unsqueeze(0).cuda()
    value = value.transpose(0, 1).unsqueeze(0).cuda()
    repeat = query.size(1) // key.size(1)
    key = key.repeat_interleave(repeat, dim=1)
    value = value.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(query.size(-1))
    causal = torch.ones(
        scores.size(-2), scores.size(-1), dtype=torch.bool, device="cuda"
    ).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
        torch.bfloat16
    )
    return torch.matmul(probabilities, value)[0].transpose(0, 1).cpu()


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
    diagnostic_layer = int(os.environ.get("LLAISYS_M6B_ATTENTION_LAYER", "3"))
    os.environ["LLAISYS_QWEN2_TRACE_LAYER"] = str(diagnostic_layer)
    model = llaisys.Qwen2(str(model_dir), device="cuda", dtype="bf16")
    trace = model.forward_trace(tokens)
    query = trace["diagnostic_q_rope"]
    key = trace["diagnostic_k_rope"]
    value = trace["diagnostic_v"].reshape(len(tokens), 2, 64)
    expected = torch_attention(query, key, value)
    query_ll, key_ll, value_ll = paired_inputs([query, key, value])["cuda"]
    output = empty(
        expected.shape, "bf16", llaisys.DeviceType.NVIDIA
    )
    llaisys.Ops.self_attention(
        output, query_ll, key_ll, value_ll, 1.0 / math.sqrt(64)
    )
    actual = fetch(output)
    mismatch = actual != expected
    delta = (actual.float() - expected.float()).abs()
    count = int(mismatch.sum())
    maximum = float(delta.max())
    print(
        f"prefix32_layer{diagnostic_layer}_attention "
        f"mismatches={count} max_abs={maximum}"
    )
    assert count == 0
    print("QWEN2_PREFIX32_ATTENTION_EXACT_PASS")


if __name__ == "__main__":
    main()
