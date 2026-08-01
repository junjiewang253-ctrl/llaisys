#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys


MESSAGE = {"role": "user", "content": "你好，请用一句话介绍你自己。"}


def stats(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mismatches_3e2": int((delta > 3e-2).sum()),
        "elements": delta.numel(),
    }


def main():
    model_dir = Path(os.environ["LLAISYS_M5_MODEL_DIR"])
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
    prefix = list(tokenizer.apply_chat_template(
        [MESSAGE], tokenize=True, add_generation_prompt=True
    ))
    model = llaisys.Qwen2(str(model_dir), device="cpu", dtype="bf16")
    model.reset_cache()
    prefill = model.forward_cached_trace(prefix)
    with torch.no_grad():
        reference_prefill = reference(
            input_ids=torch.tensor([prefix], dtype=torch.int64),
            use_cache=True,
            return_dict=True,
        )
    token = int(torch.argmax(reference_prefill.logits[0, -1]))
    cached = model.forward_cached_trace([token])
    full = model.forward_trace(prefix + [token])
    with torch.no_grad():
        reference_decode = reference(
            input_ids=torch.tensor([[token]], dtype=torch.int64),
            past_key_values=reference_prefill.past_key_values,
            use_cache=True,
            return_dict=True,
        )
    report = {
        "prefix_length": len(prefix),
        "decode_token": token,
        "cursor": model.cache_info()["cursor"],
        "embedding": stats(cached["embedding"][0], full["embedding"][-1]),
    }
    for layer in range(int(model.meta.nlayer)):
        report[f"layer.{layer}.attention_out"] = stats(
            cached[f"layer.{layer}.attention_out"][0],
            full[f"layer.{layer}.attention_out"][-1],
        )
        report[f"layer.{layer}.output"] = stats(
            cached[f"layer.{layer}.output"][0],
            full[f"layer.{layer}.output"][-1],
        )
    report["final_norm"] = stats(
        cached["final_norm"][0], full["final_norm"][-1]
    )
    report["logits"] = stats(cached["logits"][0], full["logits"][-1])
    report["cached_vs_past_logits"] = stats(
        cached["logits"][0], reference_decode.logits[0, -1]
    )
    report["full_vs_past_logits"] = stats(
        full["logits"][-1], reference_decode.logits[0, -1]
    )
    full_logits = full["logits"][-1].float()
    past_logits = reference_decode.logits[0, -1].float()
    full_radius = 3e-2 + 3e-2 * full_logits.abs()
    past_radius = 3e-2 + 3e-2 * past_logits.abs()
    disjoint = (
        torch.maximum(full_logits - full_radius, past_logits - past_radius)
        > torch.minimum(full_logits + full_radius, past_logits + past_radius)
    )
    report["frozen_interval_intersection"] = {
        "disjoint_elements": int(disjoint.sum()),
        "elements": int(disjoint.numel()),
        "atol": 3e-2,
        "rtol": 3e-2,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
