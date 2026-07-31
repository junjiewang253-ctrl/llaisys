#!/usr/bin/env python3
import json
import os
import random
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Qwen2Config, Qwen2ForCausalLM

import llaisys


SEED = 20260731
FIXED_PREFIXES = [
    [1],
    [1, 5, 7, 9],
    [31, 0, 2],
    [1, 1, 1, 1, 1],
    [3, 8, 13, 21, 2, 5, 7, 11],
]


def oracle(reference, tokens):
    trace = {}
    hooks = [
        reference.model.embed_tokens.register_forward_hook(
            lambda module, inputs, output: trace.__setitem__("embedding", output[0])
        ),
        reference.model.norm.register_forward_hook(
            lambda module, inputs, output: trace.__setitem__("final_norm", output[0])
        ),
    ]
    for layer_index, layer in enumerate(reference.model.layers):
        hooks.append(
            layer.self_attn.register_forward_hook(
                lambda module, inputs, output, i=layer_index: trace.__setitem__(
                    f"layer.{i}.attention_out", output[0][0]
                )
            )
        )
        hooks.append(
            layer.register_forward_hook(
                lambda module, inputs, output, i=layer_index: trace.__setitem__(
                    f"layer.{i}.output", output[0][0]
                )
            )
        )
    with torch.no_grad():
        result = reference(
            torch.tensor([tokens], dtype=torch.int64),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    for hook in hooks:
        hook.remove()
    logits = result.logits[0]
    trace["logits"] = logits
    values = torch.topk(logits[-1].float(), 2).values
    trace["greedy_token"] = int(torch.argmax(logits[-1]))
    return trace, float(values[0] - values[1])


def main():
    model_dir = Path(os.environ["LLAISYS_M4_TINY_DIR"])
    manifest = (model_dir / "tensor-manifest.tsv").read_text().splitlines()
    assert len(manifest) == 28
    assert sum(int(row.rsplit("\t", 1)[1]) for row in manifest[1:]) == 23104

    config = Qwen2Config.from_json_file(str(model_dir / "config.json"))
    reference = Qwen2ForCausalLM(config).eval()
    missing, unexpected = reference.load_state_dict(
        load_file(model_dir / "model.safetensors"), strict=True
    )
    assert not missing and not unexpected
    model = llaisys.Qwen2(str(model_dir), device="cpu", dtype="f32")
    rng = random.Random(SEED)
    prefixes = list(FIXED_PREFIXES)
    prefixes.extend(
        [rng.randint(0, 31) for _ in range(rng.randint(1, 8))]
        for _ in range(16)
    )
    max_error = {}
    for tokens in prefixes:
        expected, margin = oracle(reference, tokens)
        assert margin >= 1e-3, (tokens, margin)
        actual = model.forward_trace(tokens)
        for name, expected_tensor in expected.items():
            if name == "greedy_token":
                assert actual[name] == expected_tensor
                continue
            tolerance = 1e-4 if name == "logits" else 5e-5
            delta = (actual[name].float() - expected_tensor.float()).abs()
            max_error[name] = max(
                max_error.get(name, 0.0),
                float(delta.max()) if delta.numel() else 0.0,
            )
            torch.testing.assert_close(
                actual[name], expected_tensor, atol=tolerance, rtol=tolerance
            )
    print(
        json.dumps(
            {
                "fixture": "Qwen2TinyFixture-v1",
                "cases": len(prefixes),
                "max_abs": max_error,
                "status": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
