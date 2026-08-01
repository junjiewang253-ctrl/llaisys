#!/usr/bin/env python3
"""Diagnostic-only M6-B sweep of every real-model CUDA layer boundary."""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys


def stats(actual, expected):
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    delta = (actual - expected).abs()
    limit = 3e-2 + 3e-2 * expected.abs()
    return {
        "elements": delta.numel(),
        "violations": int((delta > limit).sum()),
        "max_abs": float(delta.max()),
        "max_rel": float((delta / expected.abs().clamp_min(1e-12)).max()),
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
    captured = {}
    hooks = []
    for layer_id, layer in enumerate(reference.model.layers):
        hooks.extend([
            layer.self_attn.register_forward_hook(
                lambda module, inputs, output, i=layer_id: captured.__setitem__(
                    f"layer.{i}.attention_out", output[0][0, -1].detach().cpu()
                )
            ),
            layer.register_forward_hook(
                lambda module, inputs, output, i=layer_id: captured.__setitem__(
                    f"layer.{i}.output", output[0][0, -1].detach().cpu()
                )
            ),
        ])
    with torch.no_grad():
        reference(
            input_ids=torch.tensor([tokens], dtype=torch.int64, device="cuda"),
            use_cache=False, return_dict=True,
        )
    for hook in hooks:
        hook.remove()
    model = llaisys.Qwen2(str(root), device="cuda", dtype="bf16")
    trace = model.forward_trace(tokens)
    result = {}
    for layer_id in range(len(reference.model.layers)):
        for boundary in ("attention_out", "output"):
            name = f"layer.{layer_id}.{boundary}"
            result[name] = stats(trace[name][-1], captured[name])
    print(json.dumps({
        "role": "diagnostic_only",
        "case": "P3-prefix-1",
        "atol": 0.03,
        "rtol": 0.03,
        "boundaries": result,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
