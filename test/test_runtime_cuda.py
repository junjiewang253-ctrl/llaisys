#!/usr/bin/env python3
import ctypes
import json

import llaisys


SIZES = (1, 4096, 1024 * 1024)


def host_bytes(pointer, size):
    return (ctypes.c_ubyte * size).from_address(int(pointer))


def fill(pointer, size, salt):
    values = host_bytes(pointer, size)
    for index in range(size):
        values[index] = (index * 131 + salt) & 0xFF
    return bytes(values)


def main():
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    assert api.get_device_count() == 1
    api.set_device(0)
    try:
        api.set_device(1)
    except ValueError:
        pass
    else:
        raise AssertionError("logical CUDA device 1 unexpectedly accepted")
    api.set_device(0)
    api.device_synchronize()

    for _ in range(256):
        stream = api.create_stream()
        assert stream
        api.stream_synchronize(stream)
        api.destroy_stream(stream)

    exact = []
    for size in SIZES:
        host_in = api.malloc_host(size)
        host_out = api.malloc_host(size)
        device_a = api.malloc_device(size)
        device_b = api.malloc_device(size)
        expected = fill(host_in, size, size & 0xFF)
        ctypes.memset(host_out, 0, size)
        api.memcpy_sync(device_a, host_in, size, llaisys.MemcpyKind.H2D)
        api.memcpy_sync(device_b, device_a, size, llaisys.MemcpyKind.D2D)
        api.memcpy_sync(host_out, device_b, size, llaisys.MemcpyKind.D2H)
        assert bytes(host_bytes(host_out, size)) == expected

        stream = api.create_stream()
        ctypes.memset(host_out, 0, size)
        api.memcpy_async(
            device_a, host_in, size, llaisys.MemcpyKind.H2D, stream
        )
        api.memcpy_async(
            device_b, device_a, size, llaisys.MemcpyKind.D2D, stream
        )
        api.memcpy_async(
            host_out, device_b, size, llaisys.MemcpyKind.D2H, stream
        )
        api.stream_synchronize(stream)
        assert bytes(host_bytes(host_out, size)) == expected
        api.destroy_stream(stream)
        api.free_device(device_b)
        api.free_device(device_a)
        api.free_host(host_out)
        api.free_host(host_in)
        exact.append(size)

    streams = [api.create_stream(), api.create_stream()]
    pairs = []
    for lane, stream in enumerate(streams):
        size = 4096
        source = api.malloc_host(size)
        target = api.malloc_host(size)
        device = api.malloc_device(size)
        expected = fill(source, size, lane + 17)
        api.memcpy_async(device, source, size, llaisys.MemcpyKind.H2D, stream)
        api.memcpy_async(target, device, size, llaisys.MemcpyKind.D2H, stream)
        pairs.append((source, target, device, expected))
    for stream in streams:
        api.stream_synchronize(stream)
    for source, target, device, expected in pairs:
        assert bytes(host_bytes(target, 4096)) == expected
        api.free_device(device)
        api.free_host(target)
        api.free_host(source)
    for stream in streams:
        api.destroy_stream(stream)

    for _ in range(256):
        pointer = api.malloc_device(4096)
        api.free_device(pointer)

    try:
        api.memcpy_sync(None, None, 1, llaisys.MemcpyKind.D2D)
    except ValueError:
        pass
    else:
        raise AssertionError("null nonzero memcpy unexpectedly accepted")

    print("RUNTIME_CUDA=" + json.dumps({
        "device_count": 1,
        "logical_device": 0,
        "stream_cycles": 256,
        "lifecycle_cycles": 256,
        "sync_async_exact_sizes": exact,
        "independent_streams": 2,
    }, sort_keys=True))
    print("M6A CUDA RUNTIME PASS")


if __name__ == "__main__":
    main()
