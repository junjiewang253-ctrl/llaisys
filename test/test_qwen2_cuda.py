#!/usr/bin/env python3
"""M6-B tiny Qwen2 four-way CUDA correctness and cache-residency gate."""

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


def load_reference(model_dir, device):
    config = Qwen2Config.from_json_file(str(model_dir / "config.json"))
    model = Qwen2ForCausalLM(config).eval()
    missing, unexpected = model.load_state_dict(
        load_file(model_dir / "model.safetensors"), strict=True
    )
    assert not missing and not unexpected
    return model.to(device)


def reference_trace(reference, tokens):
    captured = {}
    hooks = [
        reference.model.embed_tokens.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__(
                "embedding", output[0].detach().cpu()
            )
        ),
        reference.model.norm.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__(
                "final_norm", output[0].detach().cpu()
            )
        ),
    ]
    for layer_index, layer in enumerate(reference.model.layers):
        hooks.extend([
            layer.self_attn.register_forward_hook(
                lambda module, inputs, output, i=layer_index: captured.__setitem__(
                    f"layer.{i}.attention_out", output[0][0].detach().cpu()
                )
            ),
            layer.register_forward_hook(
                lambda module, inputs, output, i=layer_index: captured.__setitem__(
                    f"layer.{i}.output", output[0][0].detach().cpu()
                )
            ),
        ])
    device = next(reference.parameters()).device
    with torch.no_grad():
        output = reference(
            torch.tensor([tokens], dtype=torch.int64, device=device),
            use_cache=False,
            return_dict=True,
        )
    for hook in hooks:
        hook.remove()
    captured["logits"] = output.logits[0].detach().cpu()
    values, indices = torch.topk(captured["logits"][-1].float(), 2)
    captured["greedy_token"] = int(indices[0])
    captured["margin"] = float(values[0] - values[1])
    return captured


def compare(label, actual, expected, tolerance, errors):
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    delta = (actual - expected).abs()
    relative = delta / expected.abs().clamp_min(1e-12)
    slot = errors.setdefault(label, {"max_abs": 0.0, "max_rel": 0.0, "cases": 0})
    slot["max_abs"] = max(slot["max_abs"], float(delta.max()) if delta.numel() else 0.0)
    slot["max_rel"] = max(slot["max_rel"], float(relative.max()) if relative.numel() else 0.0)
    slot["cases"] += 1
    torch.testing.assert_close(
        actual, expected, atol=tolerance, rtol=tolerance, check_dtype=False,
        msg=lambda message: f"{label}: {message}",
    )


def compare_trace(label, actual, expected, errors):
    for name, expected_tensor in expected.items():
        if name in ("greedy_token", "margin"):
            continue
        tolerance = 2e-4 if name == "logits" else 5e-5
        compare(f"{label}:{name}", actual[name], expected_tensor, tolerance, errors)


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

    model_dir = Path(os.environ["LLAISYS_M6B_TINY_DIR"])
    manifest = (model_dir / "tensor-manifest.tsv").read_text().splitlines()
    assert len(manifest) == 28
    assert sum(int(row.rsplit("\t", 1)[1]) for row in manifest[1:]) == 23104

    reference_cpu = load_reference(model_dir, "cpu")
    reference_cuda = load_reference(model_dir, "cuda")
    cpu = llaisys.Qwen2(str(model_dir), device="cpu", dtype="f32")
    cuda = llaisys.Qwen2(str(model_dir), device="cuda", dtype="f32")
    assert cuda.device == llaisys.DeviceType.NVIDIA and cuda.device_id == 0

    rng = random.Random(SEED)
    prefixes = list(FIXED_PREFIXES)
    prefixes.extend(
        [rng.randint(0, 31) for _ in range(rng.randint(1, 8))]
        for _ in range(16)
    )
    errors = {}
    alignment = []
    for case, tokens in enumerate(prefixes):
        torch_cpu = reference_trace(reference_cpu, tokens)
        torch_cuda = reference_trace(reference_cuda, tokens)
        cpu_no_cache = cpu.forward_trace(tokens)
        cuda_no_cache = cuda.forward_trace(tokens)
        cpu.reset_cache()
        cuda.reset_cache()
        cpu_cache = cpu.forward_cached_trace(tokens)
        cuda_cache = cuda.forward_cached_trace(tokens)

        compare_trace(f"case-{case}:torch-cuda-vs-cpu", torch_cuda, torch_cpu, errors)
        compare_trace(f"case-{case}:llaisys-cpu", cpu_no_cache, torch_cpu, errors)
        compare_trace(f"case-{case}:llaisys-cuda", cuda_no_cache, torch_cuda, errors)
        compare_trace(f"case-{case}:cpu-cache", cpu_cache, torch_cpu, errors)
        compare_trace(f"case-{case}:cuda-cache", cuda_cache, torch_cuda, errors)
        tokens_seen = {
            torch_cpu["greedy_token"], torch_cuda["greedy_token"],
            cpu_no_cache["greedy_token"], cuda_no_cache["greedy_token"],
            cpu_cache["greedy_token"], cuda_cache["greedy_token"],
        }
        assert torch_cpu["margin"] >= 1e-3 and len(tokens_seen) == 1
        assert cpu.cache_info()["cursor"] == len(tokens)
        assert cuda.cache_info()["cursor"] == len(tokens)
        alignment.append({
            "case": case, "tokens": tokens,
            "greedy_token": torch_cpu["greedy_token"],
            "margin": torch_cpu["margin"],
        })

    # Incremental decode, reset/reuse, and two-instance isolation on CUDA.
    sequence = [1, 5, 7, 9, 3, 8, 13, 2]
    for length in range(1, 9):
        cuda.reset_cache()
        result = None
        for token in sequence[:length]:
            result = cuda.forward_cached_trace([token])
        expected = cuda.forward_trace(sequence[:length])
        compare(
            f"incremental-{length}", result["logits"][-1],
            expected["logits"][-1], 2e-4, errors,
        )
        assert result["greedy_token"] == expected["greedy_token"]

    info = cuda.cache_info()
    assert info["capacity"] == 16 and info["allocated_bytes"] == 2048
    addresses = (tuple(info["k_addresses"]), tuple(info["v_addresses"]))
    baseline = None
    for iteration in range(4):
        cuda.reset_cache()
        result = cuda.forward_cached_trace(sequence)
        current = cuda.cache_info()
        assert (tuple(current["k_addresses"]), tuple(current["v_addresses"])) == addresses
        if baseline is None:
            baseline = result["logits"].clone()
        else:
            assert torch.equal(result["logits"], baseline), iteration

    other = llaisys.Qwen2(str(model_dir), device="cuda", dtype="f32")
    other_info = other.cache_info()
    assert set(addresses[0] + addresses[1]).isdisjoint(
        other_info["k_addresses"] + other_info["v_addresses"]
    )
    cuda.reset_cache()
    other.reset_cache()
    for left, right in zip(sequence[:4], reversed(sequence[-4:])):
        cuda.forward_cached_trace([left])
        other.forward_cached_trace([right])
    assert cuda.cache_info()["cursor"] == other.cache_info()["cursor"] == 4

    print("LAYERWISE_ERROR=" + json.dumps(errors, sort_keys=True))
    print("TOKEN_ALIGNMENT=" + json.dumps(alignment, sort_keys=True))
    print("CACHE_BYTES=" + json.dumps({
        "theoretical": 2048, "actual": info["allocated_bytes"],
        "capacity": info["capacity"],
    }, sort_keys=True))
    print("HBM=" + json.dumps({
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }, sort_keys=True))
    print("M6B TINY FOUR WAY PASS")


if __name__ == "__main__":
    main()
