#!/usr/bin/env python3
"""Diagnostic-only real-model CUDA boundary breakdown for Qwen2 layer 23."""

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

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
    prefix_length = int(os.environ.get("LLAISYS_M6B_DIAGNOSTIC_PREFIX", "1"))
    tokens = list(tokenizer.apply_chat_template(
        [message], tokenize=True, add_generation_prompt=True
    ))[:prefix_length]
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
    head_dim = reference.config.hidden_size // reference.config.num_attention_heads
    query = captured["q"].view(
        1, len(tokens), reference.config.num_attention_heads, head_dim
    ).transpose(1, 2)
    key = captured["k"].view(
        1, len(tokens), reference.config.num_key_value_heads, head_dim
    ).transpose(1, 2)
    value = captured["v"].view(
        1, len(tokens), reference.config.num_key_value_heads, head_dim
    ).transpose(1, 2)
    position_ids = torch.arange(len(tokens), device="cuda").unsqueeze(0)
    cos, sin = layer.self_attn.rotary_emb(value, seq_len=len(tokens))
    query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
    captured["q_rope"] = query[0].transpose(0, 1)
    captured["k_rope"] = key[0].transpose(0, 1)
    os.environ["LLAISYS_QWEN2_TRACE_LAYER"] = str(layer_id)
    model = llaisys.Qwen2(str(root), device="cuda", dtype="bf16")
    trace = model.forward_trace(tokens)
    weights = load_file(root / "model.safetensors")
    layer_input = (
        trace["embedding"]
        if layer_id == 0
        else trace[f"layer.{layer_id - 1}.output"]
    )
    post_attention = trace["diagnostic_post_attention"]

    def cuda_rms_formula(source, weight_name):
        source_cuda = source.cuda()
        weight_cuda = weights[weight_name].cuda()
        normalized = source_cuda.float() * torch.rsqrt(
            source_cuda.float().pow(2).mean(-1, keepdim=True) + 1e-6
        )
        return (normalized.to(torch.bfloat16) * weight_cuda).cpu()

    mapping = {
        "attn_norm": "diagnostic_attn_norm",
        "q": "diagnostic_q",
        "k": "diagnostic_k",
        "v": "diagnostic_v",
        "q_rope": "diagnostic_q_rope",
        "k_rope": "diagnostic_k_rope",
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
    result["attn_norm_same_input_cuda_formula"] = stats(
        trace["diagnostic_attn_norm"],
        cuda_rms_formula(
            layer_input, f"model.layers.{layer_id}.input_layernorm.weight"
        ),
    )
    result["mlp_norm_same_input_cuda_formula"] = stats(
        trace["diagnostic_mlp_norm"],
        cuda_rms_formula(
            post_attention,
            f"model.layers.{layer_id}.post_attention_layernorm.weight",
        ),
    )
    print(json.dumps({
        "role": "diagnostic_only",
        "case": f"P3-prefix-{len(tokens)}",
        "layer": layer_id,
        "atol": 0.03,
        "rtol": 0.03,
        "boundaries": result,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
