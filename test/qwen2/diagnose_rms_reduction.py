#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys
from qwen2.test_real_model import PROMPTS, reference_trace


def bits(value):
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def cascade_sum_squares(row, vector_width=16):
    values = row.float().numpy().astype(np.float32, copy=False)
    vectors = np.float32(values.reshape(-1, vector_width) ** 2)
    size = len(vectors)
    size_ilp = size // 4
    level_power = max(4, (size_ilp - 1).bit_length() // 4)
    level_step = 1 << level_power
    level_mask = level_step - 1
    levels = np.zeros((4, 4, vector_width), dtype=np.float32)
    index = 0
    while index + level_step <= size_ilp:
        for _ in range(level_step):
            for partial in range(4):
                levels[0, partial] = np.float32(
                    levels[0, partial] + vectors[index * 4 + partial]
                )
            index += 1
        for level in range(1, 4):
            for partial in range(4):
                levels[level, partial] = np.float32(
                    levels[level, partial] + levels[level - 1, partial]
                )
                levels[level - 1, partial].fill(np.float32(0))
            if index & (level_mask << (level * level_power)):
                break
    while index < size_ilp:
        for partial in range(4):
            levels[0, partial] = np.float32(
                levels[0, partial] + vectors[index * 4 + partial]
            )
        index += 1
    for level in range(1, 4):
        for partial in range(4):
            levels[0, partial] = np.float32(
                levels[0, partial] + levels[level, partial]
            )
    accumulator = np.float32(levels[0, 0] + levels[0, 1])
    accumulator = np.float32(accumulator + levels[0, 2])
    accumulator = np.float32(accumulator + levels[0, 3])
    result = np.float32(0)
    for value in accumulator:
        result = np.float32(result + value)
    return result


def main():
    model_dir = Path(os.environ["LLAISYS_M4_MODEL_DIR"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokens = tokenizer.apply_chat_template(
        [PROMPTS["P2"]], tokenize=True, add_generation_prompt=True
    )
    reference = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    backend = llaisys.Qwen2(str(model_dir), device="cpu", dtype="bf16")
    expected = reference_trace(reference, tokens, [5])
    os.environ["LLAISYS_QWEN2_TRACE_LAYER"] = "6"
    actual = backend.forward_trace(tokens)
    source = actual["layer.5.output"]
    assert torch.equal(source, expected["layer.5.output"])
    weight = load_file(model_dir / "model.safetensors")[
        "model.layers.6.input_layernorm.weight"
    ]
    torch_output = (
        source.float()
        * torch.rsqrt(source.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16) * weight
    rows = []
    candidate_output = torch.empty_like(source)
    for row_index, row in enumerate(source):
        candidate_sum = cascade_sum_squares(row, 16)
        candidate_sum_avx2 = cascade_sum_squares(row, 8)
        candidate_mean = np.float32(candidate_sum / np.float32(row.numel()))
        candidate_inverse = np.float32(
            np.float32(1.0)
            / np.sqrt(np.float32(candidate_mean + np.float32(1e-6)))
        )
        candidate_output[row_index] = (
            torch.from_numpy(
                np.asarray(
                    row.float().numpy() * candidate_inverse,
                    dtype=np.float32,
                )
            ).to(torch.bfloat16)
            * weight
        )
        torch_sum = row.float().pow(2).sum()
        if not torch.equal(
            actual["diagnostic_attn_norm"][row_index],
            torch_output[row_index],
        ):
            rows.append(
                {
                    "row": row_index,
                    "candidate_sum": float(candidate_sum),
                    "candidate_sum_bits": bits(candidate_sum),
                    "candidate_sum_avx2": float(candidate_sum_avx2),
                    "candidate_sum_avx2_bits": bits(candidate_sum_avx2),
                    "torch_sum": float(torch_sum),
                    "torch_sum_bits": bits(torch_sum),
                }
            )
    print(
        json.dumps(
            {
                "tokens": len(tokens),
                "mismatched_rows": rows,
                "backend_vs_candidate": int(
                    (actual["diagnostic_attn_norm"] != candidate_output).sum()
                ),
                "backend_vs_torch": int(
                    (actual["diagnostic_attn_norm"] != torch_output).sum()
                ),
                "candidate_vs_torch": int(
                    (candidate_output != torch_output).sum()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
