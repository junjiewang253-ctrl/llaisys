#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "torch_dtype": "float32",
    "vocab_size": 32,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "max_position_embeddings": 16,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "attention_dropout": 0.0,
    "attention_bias": True,
    "tie_word_embeddings": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


def values(shape, ordinal):
    count = 1
    for dim in shape:
        count *= dim
    index = torch.arange(count, dtype=torch.float32)
    value = ((index * (ordinal * 2 + 3) + ordinal * 11) % 101 - 50) / 96
    return value.reshape(shape).contiguous()


def make_weights():
    shapes = {
        "model.embed_tokens.weight": (32, 16),
        "lm_head.weight": (32, 16),
        "model.norm.weight": (16,),
    }
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (16,),
                prefix + "self_attn.q_proj.weight": (16, 16),
                prefix + "self_attn.q_proj.bias": (16,),
                prefix + "self_attn.k_proj.weight": (8, 16),
                prefix + "self_attn.k_proj.bias": (8,),
                prefix + "self_attn.v_proj.weight": (8, 16),
                prefix + "self_attn.v_proj.bias": (8,),
                prefix + "self_attn.o_proj.weight": (16, 16),
                prefix + "post_attention_layernorm.weight": (16,),
                prefix + "mlp.gate_proj.weight": (32, 16),
                prefix + "mlp.up_proj.weight": (32, 16),
                prefix + "mlp.down_proj.weight": (16, 32),
            }
        )
    assert len(shapes) == 27
    return {
        name: values(shape, ordinal)
        for ordinal, (name, shape) in enumerate(shapes.items(), start=1)
    }


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    weights = make_weights()
    weight_path = output / "model.safetensors"
    save_file(weights, weight_path)
    (output / "config.json").write_text(
        json.dumps(CONFIG, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    rows = ["name\tshape\tdtype\tnbytes"]
    raw_bytes = 0
    for name in sorted(weights):
        tensor = weights[name]
        raw_bytes += tensor.numel() * tensor.element_size()
        rows.append(
            f"{name}\t{list(tensor.shape)}\t{tensor.dtype}\t"
            f"{tensor.numel() * tensor.element_size()}"
        )
    assert raw_bytes == 23104
    (output / "tensor-manifest.tsv").write_text("\n".join(rows) + "\n")
    for path in (weight_path, output / "config.json", output / "tensor-manifest.tsv"):
        print(path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
