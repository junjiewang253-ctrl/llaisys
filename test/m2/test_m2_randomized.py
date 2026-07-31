#!/usr/bin/env python3
import ctypes
import random
import struct

from m2.test_m2_abi import load_lib, make_tensor, require_success
from m2.test_m2_tensor import values_f32


SEED = 20260731


def test_randomized() -> None:
    rng = random.Random(SEED)
    lib = load_lib()
    for case in range(64):
        a = rng.randint(1, 4)
        b = rng.randint(1, 4)
        c = rng.randint(1, 4)
        count = a * b * c
        base = make_tensor(lib, (a, b, c))
        values = [float(case * 1000 + i) for i in range(count)]
        payload = struct.pack(f"={count}f", *values)
        source = ctypes.create_string_buffer(payload)
        lib.tensorLoad(base, source)
        require_success(lib)

        order_tuple = rng.choice(((0, 1, 2), (1, 0, 2), (2, 1, 0)))
        order = (ctypes.c_size_t * 3)(*order_tuple)
        view = lib.tensorPermute(base, order)
        require_success(lib)
        got = values_f32(lib, view)
        expected = []
        dims = (a, b, c)
        out_shape = tuple(dims[i] for i in order_tuple)
        inverse = [order_tuple.index(i) for i in range(3)]
        for i in range(out_shape[0]):
            for j in range(out_shape[1]):
                for k in range(out_shape[2]):
                    out_index = (i, j, k)
                    original = tuple(out_index[inverse[x]] for x in range(3))
                    expected.append(
                        values[(original[0] * b + original[1]) * c + original[2]]
                    )
        assert got == expected
        lib.tensorDestroy(view)
        lib.tensorDestroy(base)
        require_success(lib)
    print(f"seed={SEED}")
    print("random_cases=64")


if __name__ == "__main__":
    test_randomized()
