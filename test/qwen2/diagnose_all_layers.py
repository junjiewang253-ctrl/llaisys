#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys
from qwen2.diagnose_real_first import errors
from qwen2.test_real_model import PROMPTS, reference_trace


def main():
    model_dir = Path(os.environ["LLAISYS_M4_MODEL_DIR"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompt_id = os.environ.get("LLAISYS_M4_PROMPT_ID", "P1")
    tokens = tokenizer.apply_chat_template(
        [PROMPTS[prompt_id]], tokenize=True, add_generation_prompt=True
    )
    reference = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    backend = llaisys.Qwen2(str(model_dir), device="cpu", dtype="bf16")
    layer_ids = list(range(len(reference.model.layers)))
    expected = reference_trace(reference, tokens, layer_ids)
    actual = backend.forward_trace(tokens)
    result = {"prompt_id": prompt_id, "tokens": len(tokens), "tensors": {}}
    for name, expected_tensor in expected.items():
        actual_tensor = actual[name]
        if name == "logits":
            actual_tensor = actual_tensor[-1]
            expected_tensor = expected_tensor[-1]
        slot = errors(actual_tensor, expected_tensor)
        slot["bitwise_mismatched"] = int(
            (
                actual_tensor.float().view(torch.int32)
                != expected_tensor.float().view(torch.int32)
            ).sum()
        )
        result["tensors"][name] = slot
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
