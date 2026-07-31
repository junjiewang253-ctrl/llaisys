#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys
from qwen2.test_real_model import PROMPTS, reference_trace


def errors(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    relative = delta / expected.float().abs().clamp_min(1e-12)
    return {
        "max_abs": float(delta.max()),
        "max_rel": float(relative.max()),
        "mismatched": int(
            (delta > (0.03 + 0.03 * expected.float().abs())).sum()
        ),
        "elements": delta.numel(),
    }


def main():
    model_dir = Path(os.environ["LLAISYS_M4_MODEL_DIR"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokens = tokenizer.apply_chat_template(
        [PROMPTS["P1"]], tokenize=True, add_generation_prompt=True
    )
    eager = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    backend = llaisys.Qwen2(str(model_dir), device="cpu", dtype="bf16")
    layer_ids = [0, len(eager.model.layers) // 2, len(eager.model.layers) - 1]
    diagnostic = {}
    layer0 = eager.model.layers[0]

    def capture_norm(module, inputs, output):
        diagnostic["diagnostic_post_attention"] = inputs[0][0]
        diagnostic["diagnostic_mlp_norm"] = output[0]

    hooks = [
        layer0.post_attention_layernorm.register_forward_hook(capture_norm),
        layer0.mlp.gate_proj.register_forward_hook(
            lambda module, inputs, output: diagnostic.__setitem__(
                "diagnostic_gate", output[0]
            )
        ),
        layer0.mlp.up_proj.register_forward_hook(
            lambda module, inputs, output: diagnostic.__setitem__(
                "diagnostic_up", output[0]
            )
        ),
        layer0.mlp.down_proj.register_forward_pre_hook(
            lambda module, inputs: diagnostic.__setitem__(
                "diagnostic_activation", inputs[0][0]
            )
        ),
        layer0.mlp.down_proj.register_forward_hook(
            lambda module, inputs, output: diagnostic.__setitem__(
                "diagnostic_mlp_out", output[0]
            )
        ),
    ]
    expected = reference_trace(eager, tokens, layer_ids)
    for hook in hooks:
        hook.remove()
    expected.update(diagnostic)
    actual = backend.forward_trace(tokens)
    norm_weight = load_file(model_dir / "model.safetensors")[
        "model.layers.0.post_attention_layernorm.weight"
    ]
    post = actual["diagnostic_post_attention"]
    normalized = post.float() * torch.rsqrt(
        post.float().pow(2).mean(-1, keepdim=True) + 1e-6
    )
    python_from_backend_input = normalized.to(torch.bfloat16) * norm_weight
    result = {
        "reference_attention_class": type(eager.model.layers[0].self_attn).__name__,
        "tensors": {},
        "rms_formula_from_backend_input": errors(
            actual["diagnostic_mlp_norm"], python_from_backend_input
        ),
    }
    for name, expected_tensor in expected.items():
        actual_tensor = actual[name]
        if name == "logits":
            actual_tensor = actual_tensor[-1]
            expected_tensor = expected_tensor[-1]
        result["tensors"][name] = errors(actual_tensor, expected_tensor)
        if actual_tensor.ndim >= 2:
            result["tensors"][name + ".last_position"] = errors(
                actual_tensor[-1], expected_tensor[-1]
            )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
