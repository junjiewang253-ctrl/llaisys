#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys


PROMPTS = {
    "P1": {"role": "user", "content": "你好，请用一句话介绍你自己。"},
    "P2": {
        "role": "user",
        "content": "What is 17 plus 25? Answer with only the number.",
    },
    "P3": {
        "role": "user",
        "content": (
            "Count from one to ten in English, separated by commas, and then "
            "explain in one sentence why ten follows nine."
        ),
    },
}


def reference_trace(reference, tokens, layer_ids):
    captured = {}
    hooks = [
        reference.model.norm.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__(
                "final_norm", output[0].detach().cpu()
            )
        )
    ]
    for layer_id in layer_ids:
        layer = reference.model.layers[layer_id]
        hooks.extend(
            [
                layer.self_attn.register_forward_hook(
                    lambda module, inputs, output, i=layer_id: captured.__setitem__(
                        f"layer.{i}.attention_out", output[0][0].detach().cpu()
                    )
                ),
                layer.register_forward_hook(
                    lambda module, inputs, output, i=layer_id: captured.__setitem__(
                        f"layer.{i}.output", output[0][0].detach().cpu()
                    )
                ),
            ]
        )
    with torch.no_grad():
        output = reference(
            input_ids=torch.tensor([tokens], dtype=torch.int64),
            use_cache=False,
            return_dict=True,
        )
    for hook in hooks:
        hook.remove()
    captured["logits"] = output.logits[0].detach().cpu()
    return captured


def compare(name, actual, expected, errors):
    delta = (actual.float() - expected.float()).abs()
    abs_error = float(delta.max()) if delta.numel() else 0.0
    relative = delta / expected.float().abs().clamp_min(1e-12)
    rel_error = float(relative.max()) if relative.numel() else 0.0
    slot = errors.setdefault(name, {"max_abs": 0.0, "max_rel": 0.0})
    slot["max_abs"] = max(slot["max_abs"], abs_error)
    slot["max_rel"] = max(slot["max_rel"], rel_error)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


def main():
    model_dir = Path(os.environ["LLAISYS_M4_MODEL_DIR"])
    tokenizer_config = (model_dir / "tokenizer_config.json").read_bytes()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    reference = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    ).eval()
    backend = llaisys.Qwen2(str(model_dir), device="cpu", dtype="bf16")
    layer_ids = [0, len(reference.model.layers) // 2, len(reference.model.layers) - 1]
    errors = {}
    alignment = {
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode("utf-8")
        ).hexdigest(),
        "tokenizer_config_sha256": hashlib.sha256(tokenizer_config).hexdigest(),
        "prompts": {},
    }

    tokenized = {}
    for prompt_id, message in PROMPTS.items():
        ids = tokenizer.apply_chat_template(
            [message], tokenize=True, add_generation_prompt=True
        )
        tokenized[prompt_id] = list(ids)
        alignment["prompts"][prompt_id] = {
            "message": message,
            "input_ids": list(ids),
            "attention_mask": [1] * len(ids),
        }
    assert len(tokenized["P3"]) >= 32

    cases = [(name, ids) for name, ids in tokenized.items()]
    cases.extend((f"P3-prefix-{length}", tokenized["P3"][:length]) for length in (1, 8, 32))
    for case_id, tokens in cases:
        expected = reference_trace(reference, tokens, layer_ids)
        actual = backend.forward_trace(tokens)
        for name, expected_tensor in expected.items():
            if name == "logits":
                compare(name, actual[name][-1], expected_tensor[-1], errors)
            else:
                compare(name, actual[name], expected_tensor, errors)
        alignment.setdefault("cases", {})[case_id] = {"prefix_length": len(tokens)}

    for prompt_id, initial in tokenized.items():
        tokens = list(initial)
        generated = []
        steps = []
        for _ in range(8):
            expected = reference_trace(reference, tokens, [])
            actual = backend.forward_trace(tokens)
            compare("generation_logits", actual["logits"][-1], expected["logits"][-1], errors)
            values, indices = torch.topk(expected["logits"][-1].float(), 2)
            oracle_token = int(indices[0])
            margin = float(values[0] - values[1])
            backend_token = int(actual["greedy_token"])
            if margin >= 5e-2:
                assert backend_token == oracle_token
            steps.append(
                {
                    "prefix_length": len(tokens),
                    "oracle_token": oracle_token,
                    "backend_token": backend_token,
                    "margin": margin,
                    "strict_top1": margin >= 5e-2,
                }
            )
            generated.append(oracle_token)
            tokens.append(oracle_token)
        alignment["prompts"][prompt_id]["generated_ids"] = generated
        alignment["prompts"][prompt_id]["steps"] = steps

    print("TOKEN_ALIGNMENT=" + json.dumps(alignment, ensure_ascii=False, sort_keys=True))
    print("LAYERWISE_ERROR=" + json.dumps(errors, sort_keys=True))
    print("M4 REAL MODEL PASS")


if __name__ == "__main__":
    main()
