#!/usr/bin/env python3
import json
import os
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


def oracle(model_dir, tokens):
    config = Qwen2Config.from_json_file(str(model_dir / "config.json"))
    reference = Qwen2ForCausalLM(config).eval()
    state = load_file(model_dir / "model.safetensors")
    missing, unexpected = reference.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    with torch.no_grad():
        result = reference(
            torch.tensor([tokens], dtype=torch.int64),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    logits = result.logits[0]
    values = torch.topk(logits[-1].float(), 2).values
    return logits, int(torch.argmax(logits[-1])), float(values[0] - values[1])


def main():
    model_dir = Path(os.environ["LLAISYS_M4_TINY_DIR"])
    manifest = (model_dir / "tensor-manifest.tsv").read_text().splitlines()
    assert len(manifest) == 28
    assert sum(int(row.rsplit("\t", 1)[1]) for row in manifest[1:]) == 23104

    model = llaisys.Qwen2(str(model_dir), device="cpu", dtype="f32")
    for tokens in FIXED_PREFIXES:
        logits, greedy, margin = oracle(model_dir, tokens)
        assert margin >= 1e-3, (tokens, margin)
        trace = model.forward_trace(tokens)
        assert torch.allclose(trace["logits"], logits, atol=1e-4, rtol=1e-4)
        assert trace["greedy_token"] == greedy
    print(json.dumps({"fixture": "Qwen2TinyFixture-v1", "status": "PASS"}))


if __name__ == "__main__":
    main()
