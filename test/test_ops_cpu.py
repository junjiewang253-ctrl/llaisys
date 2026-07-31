#!/usr/bin/env python3
import json
import math
import random

import torch

import llaisys
from llaisys.libllaisys import LIB_LLAISYS, check_last_error


SEED = 20260731
DTYPES = {
    "f32": (torch.float32, llaisys.DataType.F32),
    "f16": (torch.float16, llaisys.DataType.F16),
    "bf16": (torch.bfloat16, llaisys.DataType.BF16),
}
TOLERANCE = {
    "f32": {"general": 2e-5, "attention": 3e-5},
    "f16": {"general": 2e-3, "attention": 3e-3},
    "bf16": {"general": 2e-2, "attention": 3e-2},
}

case_counts = {}
max_error = {}


def count(op, category):
    case_counts.setdefault(op, {}).setdefault(category, 0)
    case_counts[op][category] += 1


def make_tensor(value: torch.Tensor) -> llaisys.Tensor:
    value = value.contiguous()
    reverse = {torch_type: llaisys_type for torch_type, llaisys_type in DTYPES.values()}
    reverse[torch.int64] = llaisys.DataType.I64
    result = llaisys.Tensor(
        tuple(value.shape), dtype=reverse[value.dtype], device=llaisys.DeviceType.CPU
    )
    result.load(value.data_ptr())
    return result


def empty(shape, dtype_name):
    return llaisys.Tensor(
        shape, dtype=DTYPES[dtype_name][1], device=llaisys.DeviceType.CPU
    )


def to_torch(value: llaisys.Tensor, dtype: torch.dtype) -> torch.Tensor:
    result = torch.empty(value.shape(), dtype=dtype)
    runtime = llaisys.RuntimeAPI(llaisys.DeviceType.CPU)
    runtime.memcpy_sync(
        result.data_ptr(),
        value.data_ptr(),
        result.numel() * result.element_size(),
        llaisys.MemcpyKind.H2H,
    )
    return result


def compare(op, dtype_name, actual, expected, tolerance=None, exact=False):
    if exact:
        assert torch.equal(actual, expected), (op, dtype_name, actual, expected)
        abs_error = 0.0
        rel_error = 0.0
    else:
        tolerance = TOLERANCE[dtype_name]["general"] if tolerance is None else tolerance
        actual_f = actual.float()
        expected_f = expected.float()
        delta = (actual_f - expected_f).abs()
        abs_error = float(delta.max()) if delta.numel() else 0.0
        denom = expected_f.abs().clamp_min(1e-12)
        rel_error = float((delta / denom).max()) if delta.numel() else 0.0
        torch.testing.assert_close(
            actual, expected, atol=tolerance, rtol=tolerance
        )
    slot = max_error.setdefault(dtype_name, {}).setdefault(
        op, {"max_abs": 0.0, "max_rel": 0.0}
    )
    slot["max_abs"] = max(slot["max_abs"], abs_error)
    slot["max_rel"] = max(slot["max_rel"], rel_error)


def deterministic_values(shape, dtype, scale=1.0):
    count_value = math.prod(shape)
    base = torch.arange(count_value, dtype=torch.float32).reshape(shape)
    return ((base % 17) - 8).mul(scale).to(dtype)


def test_rearrange(dtype_name):
    dtype = DTYPES[dtype_name][0]
    fixed = [
        deterministic_values((2, 3, 4), dtype),
        deterministic_values((2, 3, 4), dtype).permute(1, 0, 2),
        deterministic_values((3, 4, 5), dtype)[1:, :, 1:4],
        torch.empty((2, 0, 3), dtype=dtype),
    ]
    for expected in fixed:
        base = make_tensor(expected.contiguous())
        if not expected.is_contiguous() and expected.ndim == 3:
            original = make_tensor(
                deterministic_values((2, 3, 4), dtype)
                if tuple(expected.shape) == (3, 2, 4)
                else deterministic_values((3, 4, 5), dtype)
            )
            source = (
                original.permute(1, 0, 2)
                if tuple(expected.shape) == (3, 2, 4)
                else original.slice(0, 1, 3).slice(2, 1, 4)
            )
        else:
            source = base
        output = empty(tuple(expected.shape), dtype_name)
        llaisys.Ops.rearrange(output, source)
        compare("rearrange", dtype_name, to_torch(output, dtype), expected, exact=True)
        count("rearrange", "fixed")
    rng = random.Random(SEED)
    for _ in range(16):
        shape = (rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4))
        original_t = deterministic_values(shape, dtype, rng.random() + 0.1)
        original = make_tensor(original_t)
        source = original.permute(1, 0, 2)
        expected = original_t.permute(1, 0, 2)
        output = empty(tuple(expected.shape), dtype_name)
        llaisys.Ops.rearrange(output, source)
        compare("rearrange", dtype_name, to_torch(output, dtype), expected, exact=True)
        count("rearrange", "random")


def test_argmax(dtype_name):
    dtype = DTYPES[dtype_name][0]
    cases = [
        torch.tensor([3.0], dtype=dtype),
        torch.tensor([-4.0, -2.0], dtype=dtype),
        torch.tensor([-9, -3, -7, -3, -8, -4, -5], dtype=dtype),
        deterministic_values((257,), dtype),
        torch.tensor([5, 5, 1], dtype=dtype),
        torch.tensor([1, 7, 7, 2], dtype=dtype),
        torch.tensor([1, 2, 5, 5], dtype=dtype),
    ]
    generator = torch.Generator().manual_seed(SEED)
    cases.extend(torch.randn((9,), generator=generator).to(dtype) for _ in range(16))
    for index, values in enumerate(cases):
        source = make_tensor(values)
        out_index = llaisys.Tensor((1,), dtype=llaisys.DataType.I64)
        out_value = empty((1,), dtype_name)
        llaisys.Ops.argmax(out_index, out_value, source)
        expected_index = torch.argmax(values).reshape(1).to(torch.int64)
        expected_value = values[expected_index].reshape(1)
        compare("argmax_value", dtype_name, to_torch(out_value, dtype), expected_value, exact=True)
        assert torch.equal(to_torch(out_index, torch.int64), expected_index)
        count("argmax", "fixed" if index < 7 else "random")


def test_embedding(dtype_name):
    dtype = DTYPES[dtype_name][0]
    fixed = [
        (1, 3, [0]),
        (7, 5, [0, 6, 3, 3]),
        (33, 4, [32, 0, 32, 1]),
    ]
    rng = random.Random(SEED)
    fixed.extend(
        (
            rng.randint(2, 12),
            rng.randint(1, 8),
            [rng.randint(0, 1) for _ in range(rng.randint(1, 8))],
        )
        for _ in range(16)
    )
    for index, (vocab, hidden, indices) in enumerate(fixed):
        indices = [value % vocab for value in indices]
        weight = deterministic_values((vocab, hidden), dtype, 0.125)
        index_t = torch.tensor(indices, dtype=torch.int64)
        output = empty((len(indices), hidden), dtype_name)
        llaisys.Ops.embedding(output, make_tensor(index_t), make_tensor(weight))
        compare(
            "embedding",
            dtype_name,
            to_torch(output, dtype),
            weight[index_t],
            exact=True,
        )
        count("embedding", "fixed" if index < 3 else "random")


def test_linear(dtype_name):
    dtype = DTYPES[dtype_name][0]
    cases = [(1, 1, 1), (2, 4, 3), (3, 31, 23)]
    rng = random.Random(SEED)
    cases.extend(
        (rng.randint(1, 4), rng.randint(1, 12), rng.randint(1, 10))
        for _ in range(16)
    )
    for index, (m, k, n) in enumerate(cases):
        for use_bias in (False, True):
            inp = deterministic_values((m, k), dtype, 0.03125)
            weight = deterministic_values((n, k), dtype, 0.015625)
            bias = deterministic_values((n,), dtype, 0.0078125) if use_bias else None
            output = empty((m, n), dtype_name)
            llaisys.Ops.linear(
                output,
                make_tensor(inp),
                make_tensor(weight),
                make_tensor(bias) if bias is not None else None,
            )
            expected = torch.nn.functional.linear(
                inp.float(),
                weight.float(),
                bias.float() if bias is not None else None,
            ).to(dtype)
            compare("linear", dtype_name, to_torch(output, dtype), expected)
            count("linear", "fixed" if index < 3 else "random")


def test_rms_norm(dtype_name):
    dtype = DTYPES[dtype_name][0]
    cases = [(1, 1), (2, 4), (3, 31), (2, 896), (1, 896)]
    rng = random.Random(SEED)
    cases.extend(
        (rng.randint(1, 4), rng.randint(1, 32)) for _ in range(16)
    )
    for index, shape in enumerate(cases):
        for eps in (1e-5, 1e-6):
            if index == 0:
                inp = torch.zeros(shape, dtype=dtype)
            elif shape[-1] == 896:
                generator = torch.Generator().manual_seed(
                    8 if shape[0] == 2 else 92
                )
                inp = torch.randn(shape, generator=generator).to(dtype)
            else:
                inp = deterministic_values(shape, dtype, 0.0625)
            weight = deterministic_values((shape[-1],), dtype, 0.03125) + 1
            output = empty(shape, dtype_name)
            llaisys.Ops.rms_norm(
                output, make_tensor(inp), make_tensor(weight), eps
            )
            normalized = inp.float() * torch.rsqrt(
                inp.float().pow(2).mean(-1, keepdim=True) + eps
            )
            if dtype != torch.float32:
                normalized = normalized.to(dtype).float()
            expected = (normalized * weight.float()).to(dtype)
            compare(
                "rms_norm",
                dtype_name,
                to_torch(output, dtype),
                expected,
                exact=dtype_name == "bf16" and shape[-1] == 896,
            )
            count("rms_norm", "fixed" if index < 3 else "random")


def rope_oracle(inp, positions, theta):
    dimension = inp.shape[-1]
    half = dimension // 2
    frequency = positions.float().unsqueeze(1) / (
        theta ** (2 * torch.arange(half, dtype=torch.float32) / dimension)
    )
    sin = frequency.sin().unsqueeze(1)
    cos = frequency.cos().unsqueeze(1)
    left = inp.float()[..., :half]
    right = inp.float()[..., half:]
    return torch.cat((left * cos - right * sin, right * cos + left * sin), -1)


def test_rope(dtype_name):
    dtype = DTYPES[dtype_name][0]
    cases = [(1, 1, 2), (3, 2, 4), (7, 4, 16)]
    rng = random.Random(SEED)
    cases.extend(
        (
            rng.randint(1, 7),
            rng.randint(1, 4),
            rng.choice((2, 4, 8, 16)),
        )
        for _ in range(16)
    )
    for index, shape in enumerate(cases):
        inp = deterministic_values(shape, dtype, 0.03125)
        positions = torch.arange(5, 5 + shape[0], dtype=torch.int64)
        output = empty(shape, dtype_name)
        llaisys.Ops.rope(
            output, make_tensor(inp), make_tensor(positions), 10000.0
        )
        expected = rope_oracle(inp, positions, 10000.0).to(dtype)
        compare("rope", dtype_name, to_torch(output, dtype), expected)
        count("rope", "fixed" if index < 3 else "random")


def test_swiglu(dtype_name):
    dtype = DTYPES[dtype_name][0]
    cases = [(1, 1), (2, 7), (3, 31)]
    rng = random.Random(SEED)
    cases.extend(
        (rng.randint(1, 4), rng.randint(1, 32)) for _ in range(16)
    )
    for index, shape in enumerate(cases):
        gate = deterministic_values(shape, dtype, 0.25)
        if index == 0:
            gate = torch.tensor([[-20.0]], dtype=dtype)
        elif index == 1:
            gate.reshape(-1)[0:3] = torch.tensor([-20, 0, 20], dtype=dtype)
        up = deterministic_values(shape, dtype, 0.125) + 1
        output = empty(shape, dtype_name)
        llaisys.Ops.swiglu(output, make_tensor(gate), make_tensor(up))
        expected = (up.float() * torch.nn.functional.silu(gate.float())).to(dtype)
        compare("swiglu", dtype_name, to_torch(output, dtype), expected)
        count("swiglu", "fixed" if index < 3 else "random")


def attention_oracle(q, k, v, scale):
    qt = q.transpose(0, 1)
    kt = k.transpose(0, 1)
    vt = v.transpose(0, 1)
    length, source = q.shape[0], k.shape[0]
    repeat = q.shape[1] // k.shape[1]
    kt = kt.repeat_interleave(repeat, 0)
    vt = vt.repeat_interleave(repeat, 0)
    scores = torch.matmul(qt, kt.transpose(-2, -1)) * scale
    mask = torch.ones((length, source), dtype=torch.bool).tril(
        diagonal=source - length
    )
    scores.masked_fill_(~mask, float("-inf"))
    probability = torch.softmax(scores, -1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probability, vt).transpose(0, 1)


def test_attention(dtype_name):
    dtype = DTYPES[dtype_name][0]
    cases = [(1, 1, 1, 1, 2), (1, 5, 4, 2, 4), (3, 5, 4, 2, 8)]
    rng = random.Random(SEED)
    cases.extend(
        (
            rng.randint(1, 4),
            rng.randint(4, 8),
            rng.choice((1, 2, 4)),
            1,
            rng.choice((2, 4, 8)),
        )
        for _ in range(16)
    )
    for index, (length, source, heads, kv_heads, dimension) in enumerate(cases):
        source = max(source, length)
        if heads % kv_heads:
            kv_heads = 1
        q = deterministic_values((length, heads, dimension), dtype, 0.03125)
        k = deterministic_values((source, kv_heads, dimension), dtype, 0.015625)
        v = deterministic_values((source, kv_heads, dimension), dtype, 0.0625)
        scale = 1.0 / math.sqrt(dimension)
        output = empty((length, heads, dimension), dtype_name)
        llaisys.Ops.self_attention(
            output, make_tensor(q), make_tensor(k), make_tensor(v), scale
        )
        expected = attention_oracle(q, k, v, scale).to(dtype)
        compare(
            "self_attention",
            dtype_name,
            to_torch(output, dtype),
            expected,
            TOLERANCE[dtype_name]["attention"],
        )
        count("self_attention", "fixed" if index < 3 else "random")


def expect_c_error(function, *args):
    function(*args)
    try:
        check_last_error()
    except (ValueError, RuntimeError, NotImplementedError):
        return
    raise AssertionError("invalid C operator call did not set an error")


def test_invalid_c_abi():
    signatures = [
        ("rearrange", LIB_LLAISYS.llaisysRearrange, [(None, None)] * 3),
        ("argmax", LIB_LLAISYS.llaisysArgmax, [(None, None, None)] * 3),
        ("embedding", LIB_LLAISYS.llaisysEmbedding, [(None, None, None)] * 3),
        ("linear", LIB_LLAISYS.llaisysLinear, [(None, None, None, None)] * 3),
        ("rms_norm", LIB_LLAISYS.llaisysRmsNorm, [(None, None, None, 1e-5)] * 3),
        ("rope", LIB_LLAISYS.llaisysROPE, [(None, None, None, 10000.0)] * 3),
        (
            "self_attention",
            LIB_LLAISYS.llaisysSelfAttention,
            [(None, None, None, None, 1.0)] * 3,
        ),
        ("swiglu", LIB_LLAISYS.llaisysSwiGLU, [(None, None, None)] * 3),
    ]
    for op, function, cases in signatures:
        for args in cases:
            expect_c_error(function, *args)
            count(op, "invalid")


def test_add_regression():
    for dtype_name, (dtype, _) in DTYPES.items():
        lhs = deterministic_values((2, 7), dtype, 0.25)
        rhs = deterministic_values((2, 7), dtype, 0.125)
        output = empty((2, 7), dtype_name)
        llaisys.Ops.add(output, make_tensor(lhs), make_tensor(rhs))
        compare("add", dtype_name, to_torch(output, dtype), lhs + rhs)


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    for dtype_name in DTYPES:
        test_rearrange(dtype_name)
        test_argmax(dtype_name)
        test_embedding(dtype_name)
        test_linear(dtype_name)
        test_rms_norm(dtype_name)
        test_rope(dtype_name)
        test_swiglu(dtype_name)
        test_attention(dtype_name)
    test_invalid_c_abi()
    test_add_regression()
    print("CASE_COUNTS=" + json.dumps(case_counts, sort_keys=True))
    print("PER_OP_ERROR=" + json.dumps(max_error, sort_keys=True))
    print("M3 CPU OPS PASS")


if __name__ == "__main__":
    main()
