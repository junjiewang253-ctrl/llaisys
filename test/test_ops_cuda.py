#!/usr/bin/env python3
import json
import math

import torch

import llaisys


SEED = 20260731
DTYPES = {
    "f32": (torch.float32, llaisys.DataType.F32, 2e-5, 3e-5),
    "f16": (torch.float16, llaisys.DataType.F16, 2e-3, 3e-3),
    "bf16": (torch.bfloat16, llaisys.DataType.BF16, 2e-2, 3e-2),
}


def llaisys_tensor(value, device):
    result = llaisys.Tensor(
        tuple(value.shape), dtype=DTYPES_BY_TORCH[value.dtype],
        device=device, device_id=0,
    )
    result.load(value.contiguous().data_ptr())
    return result


def llaisys_i64(value, device):
    result = llaisys.Tensor(
        tuple(value.shape), dtype=llaisys.DataType.I64,
        device=device, device_id=0,
    )
    result.load(value.contiguous().data_ptr())
    return result


def empty(shape, dtype_name, device):
    return llaisys.Tensor(
        shape, dtype=DTYPES[dtype_name][1], device=device, device_id=0
    )


def fetch(value):
    torch_dtype = TORCH_BY_LLAISYS[value.dtype()]
    result = torch.empty(tuple(value.shape()), dtype=torch_dtype)
    api = llaisys.RuntimeAPI(value.device_type())
    kind = (
        llaisys.MemcpyKind.D2H
        if value.device_type() == llaisys.DeviceType.NVIDIA
        else llaisys.MemcpyKind.H2H
    )
    api.memcpy_sync(
        result.data_ptr(), value.data_ptr(),
        result.numel() * result.element_size(), kind,
    )
    return result


DTYPES_BY_TORCH = {spec[0]: spec[1] for spec in DTYPES.values()}
TORCH_BY_LLAISYS = {spec[1]: spec[0] for spec in DTYPES.values()}
TORCH_BY_LLAISYS[llaisys.DataType.I64] = torch.int64


def compare(label, actual, expected, atol, stats, exact=False):
    actual = actual.cpu()
    expected = expected.cpu()
    delta = (actual.float() - expected.float()).abs()
    max_abs = float(delta.max()) if delta.numel() else 0.0
    relative = delta / expected.float().abs().clamp_min(1e-12)
    max_rel = float(relative.max()) if relative.numel() else 0.0
    slot = stats.setdefault(label, {"max_abs": 0.0, "max_rel": 0.0, "cases": 0})
    slot["max_abs"] = max(slot["max_abs"], max_abs)
    slot["max_rel"] = max(slot["max_rel"], max_rel)
    slot["cases"] += 1
    if exact:
        assert torch.equal(actual, expected), label
    else:
        torch.testing.assert_close(
            actual, expected, atol=atol, rtol=atol, check_dtype=True,
            msg=lambda message: f"{label}: {message}",
        )


def paired_inputs(values):
    result = {}
    for name, device in (("cpu", llaisys.DeviceType.CPU),
                         ("cuda", llaisys.DeviceType.NVIDIA)):
        result[name] = [
            llaisys_i64(value, device) if value.dtype == torch.int64
            else llaisys_tensor(value, device)
            for value in values
        ]
    return result


def torch_attention(q, k, v, scale):
    query = q.transpose(-2, -3)
    key = k.transpose(-2, -3)
    value = v.transpose(-2, -3)
    length, total = query.size(-2), key.size(-2)
    bias = torch.zeros(length, total, dtype=query.dtype, device=query.device)
    mask = torch.ones(length, total, dtype=torch.bool, device=query.device).tril(
        diagonal=total - length
    )
    bias.masked_fill_(mask.logical_not(), float("-inf"))
    repeat = query.size(-3) // key.size(-3)
    key = key.repeat_interleave(repeat, -3)
    value = value.repeat_interleave(repeat, -3)
    weights = torch.softmax(query @ key.transpose(-2, -1) * scale + bias, -1)
    return (weights @ value).transpose(-2, -3).contiguous()


def run_case(dtype_name, case, stats, counts):
    torch_dtype, _, tolerance, attention_tolerance = DTYPES[dtype_name]
    generator = torch.Generator().manual_seed(SEED + case)
    m, k, n = 1 + case % 3, 2 + case % 7, 2 + (case * 3) % 7
    shape = (m, k)
    a = torch.rand(shape, dtype=torch_dtype, generator=generator) - 0.5
    b = torch.rand(shape, dtype=torch_dtype, generator=generator) - 0.5
    values = paired_inputs([a, b])
    for path, device in (("cpu", "cpu"), ("cuda", "cuda")):
        out = empty(shape, dtype_name, values[path][0].device_type())
        llaisys.Ops.add(out, values[path][0], values[path][1])
        oracle = a + b if path == "cpu" else a.cuda() + b.cuda()
        compare(f"add:{dtype_name}:{path}", fetch(out), oracle, 1e-6, stats)
    counts["add"] += 1

    base = torch.rand((2, 3, 4), dtype=torch_dtype, generator=generator)
    paired = paired_inputs([base])
    for path in ("cpu", "cuda"):
        source = paired[path][0].permute(1, 0, 2)
        out = empty((3, 2, 4), dtype_name, source.device_type())
        llaisys.Ops.rearrange(out, source)
        compare(f"rearrange:{dtype_name}:{path}", fetch(out),
                base.permute(1, 0, 2), 0.0, stats, exact=True)
    counts["rearrange"] += 1

    vector = torch.rand((7 + case,), dtype=torch_dtype, generator=generator) - 0.5
    vector[case % vector.numel()] = 2
    paired = paired_inputs([vector])
    for path in ("cpu", "cuda"):
        idx = llaisys.Tensor((1,), dtype=llaisys.DataType.I64,
                             device=paired[path][0].device_type(), device_id=0)
        val = empty((1,), dtype_name, paired[path][0].device_type())
        llaisys.Ops.argmax(idx, val, paired[path][0])
        expected_val, expected_idx = torch.max(vector, dim=0, keepdim=True)
        compare(f"argmax-value:{dtype_name}:{path}", fetch(val), expected_val,
                0.0, stats, exact=True)
        compare(f"argmax-index:{dtype_name}:{path}", fetch(idx), expected_idx,
                0.0, stats, exact=True)
    counts["argmax"] += 1

    weight = torch.rand((11, n), dtype=torch_dtype, generator=generator) - 0.5
    index = torch.tensor([0, 10, case % 11, case % 11], dtype=torch.int64)
    paired = paired_inputs([index, weight])
    for path in ("cpu", "cuda"):
        out = empty((4, n), dtype_name, paired[path][1].device_type())
        llaisys.Ops.embedding(out, paired[path][0], paired[path][1])
        compare(f"embedding:{dtype_name}:{path}", fetch(out), weight[index],
                0.0, stats, exact=True)
    counts["embedding"] += 1

    x = (torch.rand((m, k), dtype=torch_dtype, generator=generator) - 0.5) * 0.1
    weight = (torch.rand((n, k), dtype=torch_dtype, generator=generator) - 0.5) * 0.1
    bias = torch.rand((n,), dtype=torch_dtype, generator=generator) - 0.5
    paired = paired_inputs([x, weight, bias])
    for path in ("cpu", "cuda"):
        out = empty((m, n), dtype_name, paired[path][0].device_type())
        llaisys.Ops.linear(out, paired[path][0], paired[path][1], paired[path][2])
        tx = x if path == "cpu" else x.cuda()
        tw = weight if path == "cpu" else weight.cuda()
        tb = bias if path == "cpu" else bias.cuda()
        compare(f"linear:{dtype_name}:{path}", fetch(out),
                torch.nn.functional.linear(tx, tw, tb), tolerance, stats)
    counts["linear"] += 1

    x = torch.rand((m, k), dtype=torch_dtype, generator=generator) - 0.5
    weight = torch.rand((k,), dtype=torch_dtype, generator=generator) + 0.5
    paired = paired_inputs([x, weight])
    for path in ("cpu", "cuda"):
        out = empty((m, k), dtype_name, paired[path][0].device_type())
        llaisys.Ops.rms_norm(out, paired[path][0], paired[path][1], 1e-5)
        tx = x if path == "cpu" else x.cuda()
        tw = weight if path == "cpu" else weight.cuda()
        oracle = tx * torch.rsqrt(tx.square().mean(-1, keepdim=True) + 1e-5) * tw
        compare(f"rms_norm:{dtype_name}:{path}", fetch(out), oracle,
                tolerance, stats)
    counts["rms_norm"] += 1

    seq, heads, dim = 1 + case % 4, 1 + case % 3, 2 * (1 + case % 4)
    x = torch.rand((seq, heads, dim), dtype=torch_dtype, generator=generator) - 0.5
    positions = torch.arange(case, case + seq, dtype=torch.int64)
    paired = paired_inputs([x, positions])
    for path in ("cpu", "cuda"):
        out = empty((seq, heads, dim), dtype_name, paired[path][0].device_type())
        llaisys.Ops.rope(out, paired[path][0], paired[path][1], 10000.0)
        tx = x if path == "cpu" else x.cuda()
        pos = positions.to(tx.device).float().unsqueeze(1)
        freq = pos / (10000.0 ** (
            2 * torch.arange(dim // 2, device=tx.device).float() / dim
        ))
        sin, cos = freq.sin().unsqueeze(1), freq.cos().unsqueeze(1)
        oracle = torch.empty_like(tx)
        oracle[..., :dim // 2] = tx[..., :dim // 2] * cos - tx[..., dim // 2:] * sin
        oracle[..., dim // 2:] = tx[..., dim // 2:] * cos + tx[..., :dim // 2] * sin
        compare(f"rope:{dtype_name}:{path}", fetch(out), oracle, tolerance, stats)
    counts["rope"] += 1

    gate = torch.rand(shape, dtype=torch_dtype, generator=generator) * 4 - 2
    up = torch.rand(shape, dtype=torch_dtype, generator=generator) - 0.5
    paired = paired_inputs([gate, up])
    for path in ("cpu", "cuda"):
        out = empty(shape, dtype_name, paired[path][0].device_type())
        llaisys.Ops.swiglu(out, paired[path][0], paired[path][1])
        tg = gate if path == "cpu" else gate.cuda()
        tu = up if path == "cpu" else up.cuda()
        compare(f"swiglu:{dtype_name}:{path}", fetch(out),
                tu * torch.nn.functional.silu(tg), tolerance, stats)
    counts["swiglu"] += 1

    qlen, total, heads, kvheads, dim = 1 + case % 3, 4 + case % 4, 4, 2, 2 * (1 + case % 3)
    total = max(total, qlen)
    q = torch.rand((qlen, heads, dim), dtype=torch_dtype, generator=generator) - 0.5
    key = torch.rand((total, kvheads, dim), dtype=torch_dtype, generator=generator) - 0.5
    value = torch.rand((total, kvheads, dim), dtype=torch_dtype, generator=generator) - 0.5
    paired = paired_inputs([q, key, value])
    scale = 1.0 / math.sqrt(dim)
    for path in ("cpu", "cuda"):
        out = empty((qlen, heads, dim), dtype_name, paired[path][0].device_type())
        llaisys.Ops.self_attention(
            out, paired[path][0], paired[path][1], paired[path][2], scale
        )
        tq = q if path == "cpu" else q.cuda()
        tk = key if path == "cpu" else key.cuda()
        tv = value if path == "cpu" else value.cuda()
        compare(f"self_attention:{dtype_name}:{path}", fetch(out),
                torch_attention(tq, tk, tv, scale), attention_tolerance, stats)
    counts["self_attention"] += 1


def main():
    assert torch.__version__ == "2.4.0+cu124"
    assert torch.version.cuda == "12.4"
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    assert torch.cuda.current_device() == 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    stats = {}
    counts = {name: 0 for name in (
        "add", "rearrange", "argmax", "embedding", "linear", "rms_norm",
        "rope", "swiglu", "self_attention",
    )}
    for dtype_name in DTYPES:
        for case in range(8):
            run_case(dtype_name, case, stats, counts)
    assert all(value == 24 for value in counts.values()), counts
    print("PER_OP_ERROR=" + json.dumps(stats, sort_keys=True))
    print("CASE_COUNTS=" + json.dumps(counts, sort_keys=True))
    print("M6A CUDA OPS FOUR WAY PASS")


if __name__ == "__main__":
    main()
