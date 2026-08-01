#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Qwen2Config, Qwen2ForCausalLM

import llaisys


SEED = 20260731
TOKENS = [1, 5, 7, 9, 3, 8, 13, 2, 6, 4, 11, 10, 12, 14, 15, 16]


def assert_logits(actual, expected, *, label):
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=1e-4, rtol=1e-4,
        msg=lambda message: f"{label}: {message}",
    )


def load_reference(model_dir):
    config = Qwen2Config.from_json_file(str(model_dir / "config.json"))
    reference = Qwen2ForCausalLM(config).eval()
    missing, unexpected = reference.load_state_dict(
        load_file(model_dir / "model.safetensors"), strict=True
    )
    assert not missing and not unexpected
    return reference


def reference_chunk(reference, chunk, past=None):
    with torch.no_grad():
        output = reference(
            input_ids=torch.tensor([chunk], dtype=torch.int64),
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
    return output.logits[0], output.past_key_values


def main():
    model_dir = Path(os.environ["LLAISYS_M5_TINY_DIR"])
    reference = load_reference(model_dir)
    no_cache = llaisys.Qwen2(str(model_dir), device="cpu", dtype="f32")
    cached = llaisys.Qwen2(str(model_dir), device="cpu", dtype="f32")

    info = cached.cache_info()
    assert info["cursor"] == 0
    assert info["capacity"] == 16
    assert info["allocated_bytes"] == 2048
    initial_addresses = (tuple(info["k_addresses"]), tuple(info["v_addresses"]))
    assert len(set(initial_addresses[0] + initial_addresses[1])) == 4

    # Prefix 1..8, one token at a time: cache, no-cache and PyTorch past agree.
    cached.reset_cache()
    past = None
    prefix_results = []
    for length in range(1, 9):
        actual = cached.forward_cached_trace([TOKENS[length - 1]])
        full = no_cache.forward_trace(TOKENS[:length])
        expected, past = reference_chunk(
            reference, [TOKENS[length - 1]], past
        )
        assert_logits(actual["logits"][-1], full["logits"][-1],
                      label=f"prefix-{length}:no-cache")
        assert_logits(actual["logits"][-1], expected[-1],
                      label=f"prefix-{length}:past")
        assert actual["greedy_token"] == full["greedy_token"]
        assert cached.cache_info()["cursor"] == length
        prefix_results.append({"length": length, "token": actual["greedy_token"]})

    # Equivalent chunked-prefill partitions.
    partition_logits = []
    for partition in ([8], [3, 5], [1, 2, 5]):
        cached.reset_cache()
        offset = 0
        result = None
        for width in partition:
            result = cached.forward_cached_trace(TOKENS[offset:offset + width])
            offset += width
        assert result is not None and offset == 8
        expected = no_cache.forward_trace(TOKENS[:8])
        assert_logits(result["logits"][-1], expected["logits"][-1],
                      label=f"partition-{partition}")
        partition_logits.append(result["logits"][-1].clone())
    for logits in partition_logits[1:]:
        assert_logits(logits, partition_logits[0], label="partition-equivalence")

    # Prefill 1/3/7 then single-token decode through length 12.
    for prefill in (1, 3, 7):
        cached.reset_cache()
        cached.forward_cached_trace(TOKENS[:prefill])
        for length in range(prefill + 1, 13):
            result = cached.forward_cached_trace([TOKENS[length - 1]])
            expected = no_cache.forward_trace(TOKENS[:length])
            assert_logits(result["logits"][-1], expected["logits"][-1],
                          label=f"prefill-{prefill}-decode-{length}")

    # Reset/reuse is deterministic and never reallocates cache payloads.
    reset_logits = None
    for iteration in range(16):
        cached.reset_cache()
        assert cached.cache_info()["cursor"] == 0
        result = cached.forward_cached_trace(TOKENS[:8])
        addresses = cached.cache_info()
        assert (tuple(addresses["k_addresses"]),
                tuple(addresses["v_addresses"])) == initial_addresses
        if reset_logits is None:
            reset_logits = result["logits"].clone()
        else:
            assert_logits(result["logits"], reset_logits,
                          label=f"reset-{iteration}")

    # No-cache API does not consume or mutate the explicit incremental state.
    cursor_before = cached.cache_info()["cursor"]
    no_cache_result = cached.forward_trace(TOKENS[:4])
    assert no_cache_result["logits"].shape[0] == 4
    assert cached.cache_info()["cursor"] == cursor_before

    # Two instances remain isolated under interleaved decode.
    other = llaisys.Qwen2(str(model_dir), device="cpu", dtype="f32")
    cached.reset_cache()
    other.reset_cache()
    for length in range(1, 9):
        left = cached.forward_cached_trace([TOKENS[length - 1]])
        right_tokens = list(reversed(TOKENS[:8]))
        right = other.forward_cached_trace([right_tokens[length - 1]])
        left_full = no_cache.forward_trace(TOKENS[:length])
        right_full = no_cache.forward_trace(right_tokens[:length])
        assert_logits(left["logits"][-1], left_full["logits"][-1],
                      label=f"isolation-left-{length}")
        assert_logits(right["logits"][-1], right_full["logits"][-1],
                      label=f"isolation-right-{length}")
    other_info = other.cache_info()
    assert set(initial_addresses[0] + initial_addresses[1]).isdisjoint(
        other_info["k_addresses"] + other_info["v_addresses"]
    )

    # Overflow is atomic; reset after a full cache restores usability.
    cached.reset_cache()
    cached.forward_cached_trace(TOKENS)
    assert cached.cache_info()["cursor"] == 16
    try:
        cached.forward_cached_trace([1])
    except RuntimeError:
        pass
    else:
        raise AssertionError("cache overflow unexpectedly succeeded")
    assert cached.cache_info()["cursor"] == 16
    cached.reset_cache()
    cached.forward_cached_trace([1])
    assert cached.cache_info()["cursor"] == 1

    print("PREFIX_ALIGNMENT=" + json.dumps(prefix_results, sort_keys=True))
    print("CACHE_BYTES=" + json.dumps({
        "theoretical": 2048,
        "actual": info["allocated_bytes"],
        "capacity": info["capacity"],
    }, sort_keys=True))
    print("RESET_ISOLATION=" + json.dumps({
        "reset_iterations": 16,
        "instances": 2,
        "addresses_stable": True,
        "addresses_disjoint": True,
    }, sort_keys=True))
    print("M5 TINY CACHE PASS")


if __name__ == "__main__":
    main()
