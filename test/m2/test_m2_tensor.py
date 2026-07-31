#!/usr/bin/env python3
import ctypes
import struct

from m2.test_m2_abi import (
    CPU,
    F32,
    INVALID_ARGUMENT,
    NOT_SUPPORTED,
    error,
    load_lib,
    make_tensor,
    require_success,
)


DTYPES = {
    2: ("?", [False, True, True, False]),
    5: ("i", [-7, 0, 9, 123456]),
    6: ("q", [-7, 0, 9, 1234567890123]),
    9: ("I", [0, 1, 9, 4_000_000_000]),
    10: ("Q", [0, 1, 9, 9_000_000_000]),
    12: ("e", [-1.5, 0.0, 2.25, 7.0]),
    13: ("f", [-1.5, 0.0, 2.25, 7.0]),
    14: ("d", [-1.5, 0.0, 2.25, 7.0]),
}


def shape_of(lib, handle):
    ndim = int(lib.tensorGetNdim(handle))
    require_success(lib)
    buf = (ctypes.c_size_t * ndim)()
    lib.tensorGetShape(handle, buf)
    require_success(lib)
    return tuple(buf)


def strides_of(lib, handle):
    ndim = int(lib.tensorGetNdim(handle))
    require_success(lib)
    buf = (ctypes.c_ssize_t * ndim)()
    lib.tensorGetStrides(handle, buf)
    require_success(lib)
    return tuple(buf)


def raw_bytes(lib, handle, nbytes):
    ptr = lib.tensorGetData(handle)
    require_success(lib)
    return ctypes.string_at(ptr, nbytes)


def load_bytes(lib, handle, payload):
    buf = ctypes.create_string_buffer(payload, max(1, len(payload)))
    lib.tensorLoad(handle, ctypes.cast(buf, ctypes.c_void_p))
    require_success(lib)


def values_f32(lib, handle):
    shape = shape_of(lib, handle)
    strides = strides_of(lib, handle)
    ptr = lib.tensorGetData(handle)
    require_success(lib)
    base = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
    out = []

    def walk(dim, offset):
        if dim == len(shape):
            out.append(float(base[offset]))
            return
        for i in range(shape[dim]):
            walk(dim + 1, offset + i * strides[dim])

    walk(0, 0)
    return out


def test_tensor() -> None:
    lib = load_lib()
    metadata = [
        (None, (), (), 1),
        ((0,), (0,), (1,), 0),
        ((1,), (1,), (1,), 1),
        ((2, 3), (2, 3), (3, 1), 6),
        ((2, 0, 3), (2, 0, 3), (0, 3, 1), 0),
        ((3, 4, 5), (3, 4, 5), (20, 5, 1), 60),
    ]
    for source_shape, expected_shape, expected_strides, _ in metadata:
        handle = make_tensor(lib, source_shape)
        assert shape_of(lib, handle) == expected_shape
        assert strides_of(lib, handle) == expected_strides
        assert lib.tensorIsContiguous(handle) == 1
        require_success(lib)
        lib.tensorDestroy(handle)
    print("M2-T04 PASS")
    print("M2-T06 PASS")

    for dtype, (fmt, values) in DTYPES.items():
        handle = make_tensor(lib, (4,), dtype)
        payload = struct.pack("=" + fmt * len(values), *values)
        load_bytes(lib, handle, payload)
        assert raw_bytes(lib, handle, len(payload)) == payload
        lib.tensorDestroy(handle)
    # BF16 exact bit patterns.
    handle = make_tensor(lib, (4,), 19)
    bf16 = struct.pack("=4H", 0xBFC0, 0, 0x4010, 0x40E0)
    load_bytes(lib, handle, bf16)
    assert raw_bytes(lib, handle, len(bf16)) == bf16
    lib.tensorDestroy(handle)
    print("M2-T05 PASS")

    base = make_tensor(lib, (2, 3, 4))
    values = list(range(24))
    load_bytes(lib, base, struct.pack("=24f", *values))

    view_shape = (ctypes.c_size_t * 2)(4, 6)
    view = lib.tensorView(base, view_shape, 2)
    require_success(lib)
    assert shape_of(lib, view) == (4, 6)
    replacement = struct.pack("=24f", *[100 + i for i in range(24)])
    load_bytes(lib, view, replacement)
    assert raw_bytes(lib, base, len(replacement)) == replacement
    print("M2-T07 PASS")

    bad_shape = (ctypes.c_size_t * 2)(5, 5)
    assert not lib.tensorView(base, bad_shape, 2)
    assert error(lib)[0] == INVALID_ARGUMENT
    print("M2-T08 PASS")

    order = (ctypes.c_size_t * 3)(1, 0, 2)
    perm = lib.tensorPermute(base, order)
    require_success(lib)
    assert shape_of(lib, perm) == (3, 2, 4)
    assert strides_of(lib, perm) == (4, 12, 1)
    assert values_f32(lib, perm)[0:5] == [100.0, 101.0, 102.0, 103.0, 112.0]
    load_bytes(lib, perm, struct.pack("=24f", *range(24)))
    assert values_f32(lib, base) != [float(i) for i in range(24)]
    print("M2-T09 PASS")

    duplicate = (ctypes.c_size_t * 3)(0, 0, 2)
    assert not lib.tensorPermute(base, duplicate)
    assert error(lib)[0] == INVALID_ARGUMENT
    print("M2-T10 PASS")

    sliced = lib.tensorSlice(base, 2, 1, 4)
    require_success(lib)
    assert shape_of(lib, sliced) == (2, 3, 3)
    nested = lib.tensorSlice(sliced, 0, 1, 2)
    require_success(lib)
    assert shape_of(lib, nested) == (1, 3, 3)
    assert values_f32(lib, nested)[0] == values_f32(lib, base)[13]
    empty = lib.tensorSlice(base, 1, 2, 2)
    require_success(lib)
    assert shape_of(lib, empty) == (2, 0, 4)
    print("M2-T11 PASS")

    assert not lib.tensorSlice(base, 4, 0, 1)
    assert error(lib)[0] == INVALID_ARGUMENT
    assert not lib.tensorSlice(base, 0, 2, 1)
    assert error(lib)[0] == INVALID_ARGUMENT
    print("M2-T12 PASS")

    survivor = lib.tensorSlice(base, 0, 0, 1)
    require_success(lib)
    lib.tensorDestroy(base)
    require_success(lib)
    assert values_f32(lib, survivor)
    print("M2-T13 PASS")

    contiguous_alias = lib.tensorContiguous(view)
    require_success(lib)
    load_bytes(lib, contiguous_alias, struct.pack("=24f", *[7.0] * 24))
    assert values_f32(lib, view) == [7.0] * 24
    dense_copy = lib.tensorContiguous(perm)
    require_success(lib)
    before = values_f32(lib, perm)
    load_bytes(lib, dense_copy, struct.pack("=24f", *[9.0] * 24))
    assert values_f32(lib, perm) == before
    assert strides_of(lib, dense_copy) == (8, 4, 1)
    print("M2-T14 PASS")

    shape_3d = (ctypes.c_size_t * 3)(2, 3, 4)
    reshaped_alias = lib.tensorReshape(view, shape_3d, 3)
    require_success(lib)
    load_bytes(lib, reshaped_alias, struct.pack("=24f", *[11.0] * 24))
    assert values_f32(lib, view) == [11.0] * 24
    reshape_copy = lib.tensorReshape(perm, shape_3d, 3)
    require_success(lib)
    before = values_f32(lib, perm)
    load_bytes(lib, reshape_copy, struct.pack("=24f", *[13.0] * 24))
    assert values_f32(lib, perm) == before
    assert not lib.tensorReshape(view, bad_shape, 2)
    assert error(lib)[0] == INVALID_ARGUMENT
    print("M2-T15 PASS")

    cpu0 = lib.tensorTo(view, CPU, 0)
    require_success(lib)
    cpudefault = lib.tensorTo(view, CPU, -1)
    require_success(lib)
    load_bytes(lib, cpu0, struct.pack("=24f", *[17.0] * 24))
    assert values_f32(lib, cpudefault) == [17.0] * 24
    assert not lib.tensorTo(view, CPU, 1)
    assert error(lib)[0] == NOT_SUPPORTED
    assert not lib.tensorTo(view, 1, 0)
    assert error(lib)[0] == NOT_SUPPORTED
    print("M2-T16 PASS")

    huge = (ctypes.c_size_t * 2)(ctypes.c_size_t(-1).value, 2)
    assert not lib.tensorCreate(huge, 2, F32, CPU, 0)
    assert error(lib)[0] == INVALID_ARGUMENT
    print("M2-T17 PASS")

    for handle in (
        view,
        perm,
        sliced,
        nested,
        empty,
        survivor,
        contiguous_alias,
        dense_copy,
        reshaped_alias,
        reshape_copy,
        cpu0,
        cpudefault,
    ):
        lib.tensorDestroy(handle)
        require_success(lib)
    print("M2-T20 PASS")


if __name__ == "__main__":
    test_tensor()
