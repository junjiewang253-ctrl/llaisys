#!/usr/bin/env python3
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

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
    prompt_id = os.environ.get("LLAISYS_M4_PROMPT_ID", "P1")
    tokens = tokenizer.apply_chat_template(
        [PROMPTS[prompt_id]], tokenize=True, add_generation_prompt=True
    )
    eager = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    backend = llaisys.Qwen2(str(model_dir), device="cpu", dtype="bf16")
    diagnostic_layer = int(os.environ.get("LLAISYS_QWEN2_TRACE_LAYER", "0"))
    sequence = len(tokens)
    layer_ids = [0, len(eager.model.layers) // 2, len(eager.model.layers) - 1]
    if diagnostic_layer not in layer_ids:
        layer_ids.append(diagnostic_layer)
    if diagnostic_layer > 0 and diagnostic_layer - 1 not in layer_ids:
        layer_ids.append(diagnostic_layer - 1)
    diagnostic = {}
    layer0 = eager.model.layers[diagnostic_layer]

    def capture_norm(module, inputs, output):
        diagnostic["diagnostic_post_attention"] = inputs[0][0]
        diagnostic["diagnostic_mlp_norm"] = output[0]

    hooks = [
        layer0.input_layernorm.register_forward_hook(
            lambda module, inputs, output: diagnostic.__setitem__(
                "diagnostic_attn_norm", output[0]
            )
        ),
        layer0.self_attn.q_proj.register_forward_hook(
            lambda module, inputs, output: diagnostic.__setitem__(
                "diagnostic_q", output[0]
            )
        ),
        layer0.self_attn.k_proj.register_forward_hook(
            lambda module, inputs, output: diagnostic.__setitem__(
                "diagnostic_k", output[0]
            )
        ),
        layer0.self_attn.v_proj.register_forward_hook(
            lambda module, inputs, output: diagnostic.__setitem__(
                "diagnostic_v", output[0]
            )
        ),
        layer0.self_attn.o_proj.register_forward_pre_hook(
            lambda module, inputs: diagnostic.__setitem__(
                "diagnostic_attn_value",
                inputs[0][0].view(
                    sequence,
                    eager.config.num_attention_heads,
                    eager.config.hidden_size
                    // eager.config.num_attention_heads,
                ),
            )
        ),
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
    query = diagnostic["diagnostic_q"].view(
        sequence, eager.config.num_attention_heads, eager.config.hidden_size
        // eager.config.num_attention_heads
    ).transpose(0, 1).unsqueeze(0)
    key = diagnostic["diagnostic_k"].view(
        sequence, eager.config.num_key_value_heads, eager.config.hidden_size
        // eager.config.num_attention_heads
    ).transpose(0, 1).unsqueeze(0)
    value = diagnostic["diagnostic_v"].view(
        sequence, eager.config.num_key_value_heads, eager.config.hidden_size
        // eager.config.num_attention_heads
    ).transpose(0, 1).unsqueeze(0)
    position_ids = torch.arange(sequence).unsqueeze(0)
    cos, sin = layer0.self_attn.rotary_emb(value, seq_len=sequence)
    query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
    diagnostic["diagnostic_q_rope"] = query[0].transpose(0, 1)
    diagnostic["diagnostic_k_rope"] = key[0].transpose(0, 1)
    repeated_key = key.repeat_interleave(
        eager.config.num_attention_heads
        // eager.config.num_key_value_heads,
        dim=1,
    )
    scores = torch.matmul(
        query, repeated_key.transpose(2, 3)
    ) / math.sqrt(eager.config.hidden_size // eager.config.num_attention_heads)
    mask = torch.ones(
        (sequence, sequence), dtype=torch.bool
    ).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    diagnostic["diagnostic_attention_scores"] = scores[0]
    diagnostic["diagnostic_attention_probabilities"] = torch.softmax(
        scores, dim=-1, dtype=torch.float32
    ).to(torch.bfloat16)[0]
    expected.update(diagnostic)
    actual = backend.forward_trace(tokens)
    norm_weight = load_file(model_dir / "model.safetensors")[
        f"model.layers.{diagnostic_layer}.post_attention_layernorm.weight"
    ]
    input_norm_weight = load_file(model_dir / "model.safetensors")[
        f"model.layers.{diagnostic_layer}.input_layernorm.weight"
    ]
    post = actual["diagnostic_post_attention"]
    normalized = post.float() * torch.rsqrt(
        post.float().pow(2).mean(-1, keepdim=True) + 1e-6
    )
    python_from_backend_input = normalized.to(torch.bfloat16) * norm_weight
    layer_input = (
        actual["embedding"]
        if diagnostic_layer == 0
        else actual[f"layer.{diagnostic_layer - 1}.output"]
    )
    input_normalized = layer_input.float() * torch.rsqrt(
        layer_input.float().pow(2).mean(-1, keepdim=True) + 1e-6
    )
    python_attn_norm_from_backend_input = (
        input_normalized.to(torch.bfloat16) * input_norm_weight
    )
    result = {
        "prompt_id": prompt_id,
        "reference_attention_class": type(eager.model.layers[0].self_attn).__name__,
        "diagnostic_layer": diagnostic_layer,
        "tensors": {},
        "rms_formula_from_backend_input": errors(
            actual["diagnostic_mlp_norm"], python_from_backend_input
        ),
        "attn_rms_formula_from_backend_input": errors(
            actual["diagnostic_attn_norm"],
            python_attn_norm_from_backend_input,
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
