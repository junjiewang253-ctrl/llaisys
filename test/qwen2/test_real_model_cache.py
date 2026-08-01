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


def reference_full(reference, tokens):
    with torch.no_grad():
        output = reference(
            input_ids=torch.tensor([tokens], dtype=torch.int64),
            use_cache=False,
            return_dict=True,
        )
    return output.logits[0].detach().cpu()


def record_cross_oracle(actual, expected, diagnostics):
    actual = actual.float()
    expected = expected.float()
    delta = (actual - expected).abs()
    actual_tol = 3e-2 + 3e-2 * actual.abs()
    expected_tol = 3e-2 + 3e-2 * expected.abs()
    lower = torch.maximum(actual - actual_tol, expected - expected_tol)
    upper = torch.minimum(actual + actual_tol, expected + expected_tol)
    diagnostics["comparisons"] += 1
    diagnostics["elements"] += delta.numel()
    diagnostics["not_close"] += int((delta > expected_tol).sum())
    diagnostics["disjoint_acceptance_intervals"] += int((lower > upper).sum())
    diagnostics["max_abs"] = max(
        diagnostics["max_abs"], float(delta.max()) if delta.numel() else 0.0
    )
    diagnostics["top_token_agree"] += int(
        int(torch.argmax(actual)) == int(torch.argmax(expected))
    )


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
    errors = {}
    cross_oracle = {
        "acceptance_role": "diagnostic_only",
        "comparisons": 0,
        "elements": 0,
        "not_close": 0,
        "disjoint_acceptance_intervals": 0,
        "max_abs": 0.0,
        "top_token_agree": 0,
    }
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

    # Mandatory fixed 1/8/32 prefill coverage against the paired oracles.
    for length in (1, 8, 32):
        prefix = tokenized["P3"][:length]
        backend.reset_cache()
        actual = backend.forward_cached_trace(prefix)
        full = backend.forward_trace(prefix)
        expected, _ = reference_chunk(reference, prefix)
        expected_full = reference_full(reference, prefix)
        compare(f"prefix-{length}:full-pair", full["logits"][-1],
                expected_full[-1], errors)
        compare(f"prefix-{length}:past", actual["logits"][-1],
                expected[-1], errors)
        record_cross_oracle(actual["logits"][-1], full["logits"][-1],
                            cross_oracle)
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
        full_reference = reference_full(reference, tokens)
        compare(f"{prompt_id}:prefill:full-pair", full["logits"][-1],
                full_reference[-1], errors)
        compare(f"{prompt_id}:prefill:past", cached["logits"][-1,
                ], past_logits[-1], errors)
        record_cross_oracle(cached["logits"][-1], full["logits"][-1],
                            cross_oracle)
        for step in range(8):
            values, indices = torch.topk(past_logits[-1].float(), 2)
            oracle_token = int(indices[0])
            margin = float(values[0] - values[1])
            tokens.append(oracle_token)
            cached = backend.forward_cached_trace([oracle_token])
            past_logits, past = reference_chunk(reference, [oracle_token], past)
            full = backend.forward_trace(tokens)
            full_reference = reference_full(reference, tokens)
            compare(f"{prompt_id}:decode-{step}:full-pair",
                    full["logits"][-1], full_reference[-1], errors)
            compare(f"{prompt_id}:decode-{step}:past",
                    cached["logits"][-1], past_logits[-1], errors)
            record_cross_oracle(cached["logits"][-1], full["logits"][-1],
                                cross_oracle)
            backend_token = int(cached["greedy_token"])
            past_values, past_indices = torch.topk(past_logits[-1].float(), 2)
            past_margin = float(past_values[0] - past_values[1])
            oracle_next = int(past_indices[0])
            if past_margin >= 5e-2:
                assert backend_token == oracle_next
            full_values, full_indices = torch.topk(full_reference[-1].float(), 2)
            full_margin = float(full_values[0] - full_values[1])
            full_oracle_next = int(full_indices[0])
            if full_margin >= 5e-2:
                assert int(full["greedy_token"]) == full_oracle_next
            alignment["prompts"][prompt_id]["steps"].append({
                "step": step,
                "input_token": oracle_token,
                "backend_token": backend_token,
                "oracle_token": oracle_next,
                "margin": past_margin,
                "strict_top1": past_margin >= 5e-2,
                "full_backend_token": int(full["greedy_token"]),
                "full_oracle_token": full_oracle_next,
                "full_margin": full_margin,
                "full_strict_top1": full_margin >= 5e-2,
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
    print("CROSS_ORACLE_DIAGNOSTIC=" + json.dumps(cross_oracle, sort_keys=True))
    print("M5 REAL MODEL CACHE PASS")


if __name__ == "__main__":
    main()
