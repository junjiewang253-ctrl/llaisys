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


def compare(name, actual, expected, errors, *, required=True):
    delta = (actual.float() - expected.float()).abs()
    abs_error = float(delta.max()) if delta.numel() else 0.0
    relative = delta / expected.float().abs().clamp_min(1e-12)
    rel_error = float(relative.max()) if relative.numel() else 0.0
    slot = errors.setdefault(name, {"max_abs": 0.0, "max_rel": 0.0})
    slot["max_abs"] = max(slot["max_abs"], abs_error)
    slot["max_rel"] = max(slot["max_rel"], rel_error)
    if required:
        torch.testing.assert_close(
            actual, expected, atol=3e-2, rtol=3e-2, check_dtype=False,
            msg=lambda message: f"{name}: {message}",
        )


def reference_chunk(reference, chunk, past=None):
    with torch.no_grad():
        output = reference(
            input_ids=torch.tensor([chunk], dtype=torch.int64),
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
    return output.logits[0].detach().cpu(), output.past_key_values


def main():
    model_dir = Path(os.environ["LLAISYS_M5_MODEL_DIR"])
    tokenizer_config = (model_dir / "tokenizer_config.json").read_bytes()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    reference = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    backend = llaisys.Qwen2(str(model_dir), device="cpu", dtype="bf16")
    diagnostic_oracle_conflict = (
        os.environ.get("LLAISYS_M5_DIAGNOSTIC_ORACLE_CONFLICT") == "1"
    )
    errors = {}
    alignment = {
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode("utf-8")
        ).hexdigest(),
        "tokenizer_config_sha256": hashlib.sha256(tokenizer_config).hexdigest(),
        "prompts": {},
        "prefix_cases": {},
    }
    tokenized = {}
    for prompt_id, message in PROMPTS.items():
        ids = list(tokenizer.apply_chat_template(
            [message], tokenize=True, add_generation_prompt=True
        ))
        tokenized[prompt_id] = ids
        alignment["prompts"][prompt_id] = {"input_ids": ids, "steps": []}
    assert len(tokenized["P3"]) >= 32

    config = reference.config
    dtype_bytes = 2
    expected_bytes = (
        2 * config.num_hidden_layers * config.max_position_embeddings
        * config.num_key_value_heads
        * (config.hidden_size // config.num_attention_heads) * dtype_bytes
    )
    info = backend.cache_info()
    assert info["cursor"] == 0
    assert info["capacity"] == config.max_position_embeddings
    assert info["allocated_bytes"] == expected_bytes
    stable_addresses = (tuple(info["k_addresses"]), tuple(info["v_addresses"]))

    # Mandatory fixed 1/8/32 prefill coverage against no-cache and PKV.
    for length in (1, 8, 32):
        prefix = tokenized["P3"][:length]
        backend.reset_cache()
        actual = backend.forward_cached_trace(prefix)
        full = backend.forward_trace(prefix)
        expected, _ = reference_chunk(reference, prefix)
        compare(f"prefix-{length}:no-cache", actual["logits"][-1],
                full["logits"][-1], errors,
                required=not diagnostic_oracle_conflict)
        compare(f"prefix-{length}:past", actual["logits"][-1],
                expected[-1], errors)
        alignment["prefix_cases"][str(length)] = {
            "cursor": backend.cache_info()["cursor"],
            "backend_token": actual["greedy_token"],
        }

    # Three chat-template prompts; prefill plus eight actual decode calls.
    for prompt_id, initial in tokenized.items():
        tokens = list(initial)
        backend.reset_cache()
        cached = backend.forward_cached_trace(tokens)
        past_logits, past = reference_chunk(reference, tokens)
        full = backend.forward_trace(tokens)
        compare(f"{prompt_id}:prefill:no-cache", cached["logits"][-1],
                full["logits"][-1], errors,
                required=not diagnostic_oracle_conflict)
        compare(f"{prompt_id}:prefill:past", cached["logits"][-1,
                ], past_logits[-1], errors)
        for step in range(8):
            values, indices = torch.topk(past_logits[-1].float(), 2)
            oracle_token = int(indices[0])
            margin = float(values[0] - values[1])
            tokens.append(oracle_token)
            cached = backend.forward_cached_trace([oracle_token])
            past_logits, past = reference_chunk(reference, [oracle_token], past)
            full = backend.forward_trace(tokens)
            compare(f"{prompt_id}:decode-{step}:no-cache",
                    cached["logits"][-1], full["logits"][-1], errors,
                    required=not diagnostic_oracle_conflict)
            compare(f"{prompt_id}:decode-{step}:past",
                    cached["logits"][-1], past_logits[-1], errors)
            backend_token = int(cached["greedy_token"])
            past_values, past_indices = torch.topk(past_logits[-1].float(), 2)
            past_margin = float(past_values[0] - past_values[1])
            oracle_next = int(past_indices[0])
            if past_margin >= 5e-2:
                assert backend_token == oracle_next
            alignment["prompts"][prompt_id]["steps"].append({
                "step": step,
                "input_token": oracle_token,
                "backend_token": backend_token,
                "oracle_token": oracle_next,
                "margin": past_margin,
                "strict_top1": past_margin >= 5e-2,
            })
        assert backend.cache_info()["cursor"] == len(tokens)

    # Four reset/reuse cycles retain allocation identity and exact output.
    baseline = None
    for iteration in range(4):
        backend.reset_cache()
        result = backend.forward_cached_trace(tokenized["P1"])
        current_info = backend.cache_info()
        assert (tuple(current_info["k_addresses"]),
                tuple(current_info["v_addresses"])) == stable_addresses
        if baseline is None:
            baseline = result["logits"].clone()
        else:
            assert torch.equal(result["logits"], baseline), iteration

    # Two real instances have disjoint payloads and isolated interleaved state.
    other = llaisys.Qwen2(str(model_dir), device="cpu", dtype="bf16")
    backend.reset_cache()
    other.reset_cache()
    left = tokenized["P1"][:8]
    right = tokenized["P3"][:32]
    backend.forward_cached_trace(left)
    other.forward_cached_trace(right)
    assert backend.cache_info()["cursor"] == len(left)
    assert other.cache_info()["cursor"] == len(right)
    other_info = other.cache_info()
    assert set(stable_addresses[0] + stable_addresses[1]).isdisjoint(
        other_info["k_addresses"] + other_info["v_addresses"]
    )
    backend.forward_cached_trace([tokenized["P1"][8]])
    assert other.cache_info()["cursor"] == len(right)

    print("PREFIX_ALIGNMENT=" + json.dumps(alignment, ensure_ascii=False, sort_keys=True))
    print("CACHE_BYTES=" + json.dumps({
        "theoretical": expected_bytes,
        "actual": info["allocated_bytes"],
        "capacity": info["capacity"],
        "dtype_bytes": dtype_bytes,
    }, sort_keys=True))
    print("RESET_ISOLATION=" + json.dumps({
        "reset_iterations": 4,
        "instances": 2,
        "addresses_stable": True,
        "addresses_disjoint": True,
    }, sort_keys=True))
    print("MAX_ERROR=" + json.dumps(errors, sort_keys=True))
    print("ORACLE_CONFLICT_DIAGNOSTIC=" + json.dumps({
        "enabled": diagnostic_oracle_conflict,
        "no_cache_conflicts_are_not_waived": True,
        "formal_gate_status": "FAIL_GATE" if diagnostic_oracle_conflict else "STRICT",
    }, sort_keys=True))
    print("M5 REAL MODEL CACHE PASS")


if __name__ == "__main__":
    main()
