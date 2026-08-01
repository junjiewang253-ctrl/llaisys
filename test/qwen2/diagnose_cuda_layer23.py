#!/usr/bin/env python3
"""Diagnostic-only real-model CUDA boundary breakdown for Qwen2 layer 23."""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys


def stats(actual, expected):
    actual = actual.detach().float().cpu().reshape(-1)
    expected = expected.detach().float().cpu().reshape(-1)
    assert actual.shape == expected.shape
    delta = (actual - expected).abs()
    limit = 3e-2 + 3e-2 * expected.abs()
    worst = int(torch.argmax(delta))
    return {
        "elements": delta.numel(),
        "violations": int((delta > limit).sum()),
        "max_abs": float(delta[worst]),
        "worst_index": worst,
        "actual_at_worst": float(actual[worst]),
        "expected_at_worst": float(expected[worst]),
    }


def main():
    torch.manual_seed(20260731)
    torch.cuda.manual_seed_all(20260731)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(os.environ["LLAISYS_M6B_MODEL_DIR"])
    tokenizer = AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
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
    ))[:1]
    reference = AutoModelForCausalLM.from_pretrained(
        root, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).eval().to("cuda")
    layer_id = int(os.environ.get("LLAISYS_M6B_DIAGNOSTIC_LAYER", "23"))
    layer = reference.model.layers[layer_id]
    captured = {}
    hooks = [
        layer.input_layernorm.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__("attn_norm", output)
        ),
        layer.self_attn.q_proj.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__("q", output)
        ),
        layer.self_attn.k_proj.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__("k", output)
        ),
        layer.self_attn.v_proj.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__("v", output)
        ),
        layer.self_attn.o_proj.register_forward_pre_hook(
            lambda module, inputs: captured.__setitem__("attn_value", inputs[0])
        ),
        layer.self_attn.o_proj.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__("attention_out", output)
        ),
        layer.post_attention_layernorm.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__("mlp_norm", output)
        ),
        layer.mlp.gate_proj.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__("gate", output)
        ),
        layer.mlp.up_proj.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__("up", output)
        ),
        layer.mlp.down_proj.register_forward_pre_hook(
            lambda module, inputs: captured.__setitem__("activation", inputs[0])
        ),
        layer.mlp.down_proj.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__("mlp_out", output)
        ),
        layer.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__("output", output[0])
        ),
    ]
    with torch.no_grad():
        reference(
            input_ids=torch.tensor([tokens], dtype=torch.int64, device="cuda"),
            use_cache=False, return_dict=True,
        )
    for hook in hooks:
        hook.remove()
    os.environ["LLAISYS_QWEN2_TRACE_LAYER"] = str(layer_id)
    model = llaisys.Qwen2(str(root), device="cuda", dtype="bf16")
    trace = model.forward_trace(tokens)
    mapping = {
        "attn_norm": "diagnostic_attn_norm",
        "q": "diagnostic_q",
        "k": "diagnostic_k",
        "v": "diagnostic_v",
        "attn_value": "diagnostic_attn_value",
        "attention_out": f"layer.{layer_id}.attention_out",
        "mlp_norm": "diagnostic_mlp_norm",
        "gate": "diagnostic_gate",
        "up": "diagnostic_up",
        "activation": "diagnostic_activation",
        "mlp_out": "diagnostic_mlp_out",
        "output": f"layer.{layer_id}.output",
    }
    result = {
        name: stats(trace[trace_name], captured[name])
        for name, trace_name in mapping.items()
    }
    print(json.dumps({
        "role": "diagnostic_only",
        "case": "P3-prefix-1",
        "layer": layer_id,
        "atol": 0.03,
        "rtol": 0.03,
        "boundaries": result,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
