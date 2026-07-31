#!/usr/bin/env python3
import ctypes
import json
from pathlib import Path
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen2.test_real_model import PROMPTS


def error(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return {
        "bitwise_mismatched": int(
            (actual.view(torch.int16) != expected.view(torch.int16)).sum()
        ),
        "elements": actual.numel(),
        "max_abs": float(delta.max()),
        "within_gate": bool(
            torch.allclose(actual, expected, atol=3e-2, rtol=3e-2)
        ),
    }


def main():
    model_dir = Path(os.environ["LLAISYS_M4_MODEL_DIR"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokens = tokenizer.apply_chat_template(
        [PROMPTS["P1"]], tokenize=True, add_generation_prompt=True
    )
    reference = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    captured = {}
    layer = reference.model.layers[0]
    hooks = [
        layer.post_attention_layernorm.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__(
                "source", output[0].detach().contiguous()
            )
        ),
        layer.mlp.gate_proj.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__(
                "expected", output[0].detach().contiguous()
            )
        ),
    ]
    with torch.no_grad():
        reference(
            input_ids=torch.tensor([tokens], dtype=torch.int64),
            use_cache=False,
            return_dict=True,
        )
    for hook in hooks:
        hook.remove()

    source = captured["source"]
    expected = captured["expected"]
    weight = layer.mlp.gate_proj.weight.detach().contiguous()
    rows, input_size = source.shape
    output_size = weight.shape[0]
    library = ctypes.CDLL(
        str(Path(torch.__file__).parent / "lib" / "libtorch_cpu.so")
    )
    gemm = library.cblas_gemm_bf16bf16f32
    gemm.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_int,
    ]

    outputs = {}
    row = torch.empty((rows, output_size), dtype=torch.float32)
    gemm(
        101, 111, 112, rows, output_size, input_size, 1.0,
        source.data_ptr(), input_size, weight.data_ptr(), input_size, 0.0,
        row.data_ptr(), output_size,
    )
    outputs["row_major"] = error(row.to(torch.bfloat16), expected)

    column = torch.empty((rows, output_size), dtype=torch.float32)
    gemm(
        102, 112, 111, output_size, rows, input_size, 1.0,
        weight.data_ptr(), input_size, source.data_ptr(), input_size, 0.0,
        column.data_ptr(), output_size,
    )
    outputs["column_major_equivalent"] = error(
        column.to(torch.bfloat16), expected
    )
    outputs["row_vs_column"] = error(
        row.to(torch.bfloat16), column.to(torch.bfloat16)
    )
    rounded = row.to(torch.bfloat16)
    mismatch_indices = (
        rounded.view(torch.int16) != expected.view(torch.int16)
    ).nonzero()
    details = []
    for row_index, output_index in mismatch_indices[:32].tolist():
        exact = torch.dot(
            source[row_index].double(), weight[output_index].double()
        )
        exact_bf16 = exact.to(torch.bfloat16)
        details.append(
            {
                "row": row_index,
                "output": output_index,
                "cblas_f32": float(row[row_index, output_index]),
                "cblas_bf16": float(rounded[row_index, output_index]),
                "exact_f64": float(exact),
                "exact_bf16": float(exact_bf16),
                "torch_bf16": float(expected[row_index, output_index]),
                "torch_matches_exact": bool(
                    expected[row_index, output_index].view(torch.int16)
                    == exact_bf16.view(torch.int16)
                ),
            }
        )
    outputs["mismatch_details"] = details
    print(json.dumps(outputs, sort_keys=True))


if __name__ == "__main__":
    main()
