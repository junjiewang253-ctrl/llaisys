#!/usr/bin/env python3
"""M6-B fixed Qwen2-0.5B CPU/CUDA no-cache/cache correctness gate."""

import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys


SEED = 20260731
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


def load_reference(model_dir, device):
    return AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().to(device)


def reference_trace(reference, tokens, layer_ids, past=None):
    captured = {}
    hooks = [
        reference.model.norm.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__(
                "final_norm", output[0][-1].detach().cpu()
            )
        )
    ]
    for layer_id in layer_ids:
        layer = reference.model.layers[layer_id]
        hooks.extend([
            layer.self_attn.register_forward_hook(
                lambda module, inputs, output, i=layer_id: captured.__setitem__(
                    f"layer.{i}.attention_out", output[0][0, -1].detach().cpu()
                )
            ),
            layer.register_forward_hook(
                lambda module, inputs, output, i=layer_id: captured.__setitem__(
                    f"layer.{i}.output", output[0][0, -1].detach().cpu()
                )
            ),
        ])
    device = next(reference.parameters()).device
    use_cache = past is not False
    past_value = None if past is False else past
    with torch.no_grad():
        result = reference(
            input_ids=torch.tensor([tokens], dtype=torch.int64, device=device),
            past_key_values=past_value,
            use_cache=use_cache,
            return_dict=True,
        )
    for hook in hooks:
        hook.remove()
    captured["logits"] = result.logits[0, -1].detach().cpu()
    values, indices = torch.topk(captured["logits"].float(), 2)
    captured["greedy_token"] = int(indices[0])
    captured["margin"] = float(values[0] - values[1])
    return captured, result.past_key_values


def last_trace(trace, layer_ids):
    result = {
        "final_norm": trace["final_norm"][-1].cpu(),
        "logits": trace["logits"][-1].cpu(),
        "greedy_token": int(trace["greedy_token"]),
    }
    for layer_id in layer_ids:
        result[f"layer.{layer_id}.attention_out"] = trace[
            f"layer.{layer_id}.attention_out"
        ][-1].cpu()
        result[f"layer.{layer_id}.output"] = trace[
            f"layer.{layer_id}.output"
        ][-1].cpu()
    return result


def compare(label, actual, expected, errors):
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    delta = (actual - expected).abs()
    relative = delta / expected.abs().clamp_min(1e-12)
    slot = errors.setdefault(label, {"max_abs": 0.0, "max_rel": 0.0, "cases": 0})
    slot["max_abs"] = max(slot["max_abs"], float(delta.max()) if delta.numel() else 0.0)
    slot["max_rel"] = max(slot["max_rel"], float(relative.max()) if relative.numel() else 0.0)
    slot["cases"] += 1
    torch.testing.assert_close(
        actual, expected, atol=3e-2, rtol=3e-2, check_dtype=False,
        msg=lambda message: f"{label}: {message}",
    )


def compare_trace(path, actual, expected, errors, layer_ids):
    for name in [
        *(f"layer.{i}.attention_out" for i in layer_ids),
        *(f"layer.{i}.output" for i in layer_ids),
        "final_norm", "logits",
    ]:
        compare(f"{path}:{name}", actual[name], expected[name], errors)
    if expected["margin"] >= 5e-2:
        assert int(actual["greedy_token"]) == int(expected["greedy_token"]), path


def record_cross(actual, expected, diagnostics):
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    delta = (actual - expected).abs()
    actual_tol = 3e-2 + 3e-2 * actual.abs()
    expected_tol = 3e-2 + 3e-2 * expected.abs()
    lower = torch.maximum(actual - actual_tol, expected - expected_tol)
    upper = torch.minimum(actual + actual_tol, expected + expected_tol)
    diagnostics["comparisons"] += 1
    diagnostics["elements"] += delta.numel()
    diagnostics["disjoint_acceptance_intervals"] += int((lower > upper).sum())
    diagnostics["max_abs"] = max(
        diagnostics["max_abs"], float(delta.max()) if delta.numel() else 0.0
    )
    diagnostics["top1_agree"] += int(
        int(torch.argmax(actual)) == int(torch.argmax(expected))
    )


def main():
    assert torch.__version__ == "2.4.0+cu124"
    assert torch.version.cuda == "12.4"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.current_device() == 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()

    model_dir = Path(os.environ["LLAISYS_M6B_MODEL_DIR"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    tokenized = {
        key: list(tokenizer.apply_chat_template(
            [message], tokenize=True, add_generation_prompt=True
        ))
        for key, message in PROMPTS.items()
    }
    assert len(tokenized["P3"]) >= 32
    cases = {
        "P3-prefix-1": tokenized["P3"][:1],
        "P3-prefix-8": tokenized["P3"][:8],
        "P3-prefix-32": tokenized["P3"][:32],
        **tokenized,
    }

    reference_cpu = load_reference(model_dir, "cpu")
    layer_ids = [0, len(reference_cpu.model.layers) // 2,
                 len(reference_cpu.model.layers) - 1]
    cpu = llaisys.Qwen2(str(model_dir), device="cpu", dtype="bf16")
    config = reference_cpu.config
    expected_cache_bytes = (
        2 * config.num_hidden_layers * config.max_position_embeddings
        * config.num_key_value_heads
        * (config.hidden_size // config.num_attention_heads) * 2
    )
    cpu_info = cpu.cache_info()
    assert cpu_info["allocated_bytes"] == expected_cache_bytes

    errors = {}
    cross_oracle = {
        "acceptance_role": "diagnostic_only",
        "comparisons": 0, "elements": 0,
        "disjoint_acceptance_intervals": 0,
        "max_abs": 0.0, "top1_agree": 0,
    }
    alignment = {
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode("utf-8")
        ).hexdigest(),
        "cases": {}, "decode": {},
    }
    cpu_oracles = {}
    for case_id, tokens in cases.items():
        expected, _ = reference_trace(reference_cpu, tokens, layer_ids, past=False)
        cpu_oracles[case_id] = expected
        no_cache = last_trace(cpu.forward_trace(tokens), layer_ids)
        cpu.reset_cache()
        cache = last_trace(cpu.forward_cached_trace(tokens), layer_ids)
        compare_trace(f"{case_id}:llaisys-cpu-no-cache", no_cache, expected, errors, layer_ids)
        compare_trace(f"{case_id}:llaisys-cpu-cache", cache, expected, errors, layer_ids)
        assert cpu.cache_info()["cursor"] == len(tokens)
        alignment["cases"][case_id] = {
            "length": len(tokens), "cpu_token": expected["greedy_token"],
            "cpu_margin": expected["margin"],
        }

    # Freeze the CPU oracle decode inputs/results before moving the reference to CUDA.
    decode_oracles = {}
    for prompt_id, initial in tokenized.items():
        tokens = list(initial)
        cpu.reset_cache()
        expected, past = reference_trace(reference_cpu, tokens, [], past=None)
        cpu.forward_cached_trace(tokens)
        steps = []
        for step in range(8):
            input_token = int(expected["greedy_token"])
            tokens.append(input_token)
            expected, past = reference_trace(reference_cpu, [input_token], [], past=past)
            cached = last_trace(cpu.forward_cached_trace([input_token]), [])
            no_cache = last_trace(cpu.forward_trace(tokens), [])
            compare_trace(f"{prompt_id}:cpu-decode-{step}:cache", cached, expected, errors, [])
            full_expected, _ = reference_trace(reference_cpu, tokens, [], past=False)
            compare_trace(f"{prompt_id}:cpu-decode-{step}:no-cache", no_cache, full_expected, errors, [])
            steps.append({
                "input_token": input_token,
                "expected": expected,
                "full_expected": full_expected,
            })
        decode_oracles[prompt_id] = steps

    del cpu
    reference_cuda = reference_cpu.to("cuda")
    del reference_cpu
    cuda = llaisys.Qwen2(str(model_dir), device="cuda", dtype="bf16")
    assert cuda.device == llaisys.DeviceType.NVIDIA and cuda.device_id == 0
    cuda_info = cuda.cache_info()
    assert cuda_info["allocated_bytes"] == expected_cache_bytes
    assert cuda_info["capacity"] == config.max_position_embeddings

    free_samples = [torch.cuda.mem_get_info()[0]]
    for case_id, tokens in cases.items():
        torch_cuda, _ = reference_trace(reference_cuda, tokens, layer_ids, past=False)
        record_cross(torch_cuda["logits"], cpu_oracles[case_id]["logits"], cross_oracle)
        no_cache = last_trace(cuda.forward_trace(tokens), layer_ids)
        cuda.reset_cache()
        cache = last_trace(cuda.forward_cached_trace(tokens), layer_ids)
        compare_trace(f"{case_id}:llaisys-cuda-no-cache", no_cache, torch_cuda, errors, layer_ids)
        compare_trace(f"{case_id}:llaisys-cuda-cache", cache, torch_cuda, errors, layer_ids)
        assert cuda.cache_info()["cursor"] == len(tokens)
        free_samples.append(torch.cuda.mem_get_info()[0])
        alignment["cases"][case_id].update({
            "cuda_token": torch_cuda["greedy_token"],
            "cuda_margin": torch_cuda["margin"],
        })

    for prompt_id, initial in tokenized.items():
        tokens = list(initial)
        cuda.reset_cache()
        _, past = reference_trace(reference_cuda, tokens, [], past=None)
        cuda.forward_cached_trace(tokens)
        records = []
        for step, oracle in enumerate(decode_oracles[prompt_id]):
            input_token = oracle["input_token"]
            tokens.append(input_token)
            torch_cuda, past = reference_trace(
                reference_cuda, [input_token], [], past=past
            )
            record_cross(torch_cuda["logits"], oracle["expected"]["logits"], cross_oracle)
            cached = last_trace(cuda.forward_cached_trace([input_token]), [])
            no_cache = last_trace(cuda.forward_trace(tokens), [])
            compare_trace(f"{prompt_id}:cuda-decode-{step}:cache", cached, torch_cuda, errors, [])
            torch_full, _ = reference_trace(reference_cuda, tokens, [], past=False)
            compare_trace(f"{prompt_id}:cuda-decode-{step}:no-cache", no_cache, torch_full, errors, [])
            records.append({
                "step": step, "input_token": input_token,
                "oracle_token": torch_cuda["greedy_token"],
                "backend_token": cached["greedy_token"],
                "margin": torch_cuda["margin"],
                "strict_top1": torch_cuda["margin"] >= 5e-2,
            })
            free_samples.append(torch.cuda.mem_get_info()[0])
        alignment["decode"][prompt_id] = records
        assert cuda.cache_info()["cursor"] == len(tokens)

    # Four deterministic reset/reuse cycles and two real CUDA instances.
    stable_addresses = (tuple(cuda_info["k_addresses"]), tuple(cuda_info["v_addresses"]))
    baseline = None
    for iteration in range(4):
        cuda.reset_cache()
        result = cuda.forward_cached_trace(tokenized["P1"])
        current = cuda.cache_info()
        assert (tuple(current["k_addresses"]), tuple(current["v_addresses"])) == stable_addresses
        if baseline is None:
            baseline = result["logits"].clone()
        else:
            assert torch.equal(result["logits"], baseline), iteration

    other = llaisys.Qwen2(str(model_dir), device="cuda", dtype="bf16")
    other_info = other.cache_info()
    assert set(stable_addresses[0] + stable_addresses[1]).isdisjoint(
        other_info["k_addresses"] + other_info["v_addresses"]
    )
    cuda.reset_cache()
    other.reset_cache()
    cuda.forward_cached_trace(tokenized["P1"][:8])
    other.forward_cached_trace(tokenized["P3"][:32])
    assert cuda.cache_info()["cursor"] == 8
    assert other.cache_info()["cursor"] == 32
    free_samples.append(torch.cuda.mem_get_info()[0])

    total_memory = torch.cuda.get_device_properties(0).total_memory
    process_peak_bytes = total_memory - min(free_samples)
    assert process_peak_bytes < 12 * 1024**3
    print("LAYERWISE_ERROR=" + json.dumps(errors, sort_keys=True))
    print("TOKEN_ALIGNMENT=" + json.dumps(alignment, ensure_ascii=False, sort_keys=True))
    print("CACHE_BYTES=" + json.dumps({
        "theoretical": expected_cache_bytes,
        "cpu_allocated": cpu_info["allocated_bytes"],
        "cuda_allocated": cuda_info["allocated_bytes"],
        "capacity": cuda_info["capacity"],
    }, sort_keys=True))
    print("HBM=" + json.dumps({
        "process_peak_used_bytes_from_mem_get_info": process_peak_bytes,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "hard_cap_bytes": 12 * 1024**3,
    }, sort_keys=True))
    print("CROSS_ORACLE_DIAGNOSTIC=" + json.dumps(cross_oracle, sort_keys=True))
    print("M6B REAL MODEL FOUR WAY CACHE PASS")


if __name__ == "__main__":
    main()
