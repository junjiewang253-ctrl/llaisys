#!/usr/bin/env python3
import ctypes
import threading

from m2.test_m2_abi import (
    CPU,
    INVALID_ARGUMENT,
    NOT_SUPPORTED,
    error,
    load_lib,
    make_tensor,
    require_success,
)


GET_COUNT = ctypes.CFUNCTYPE(ctypes.c_int)
SET_DEVICE = ctypes.CFUNCTYPE(None, ctypes.c_int)
SYNC = ctypes.CFUNCTYPE(None)
CREATE_STREAM = ctypes.CFUNCTYPE(ctypes.c_void_p)
STREAM_OP = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
MALLOC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
FREE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
MEMCPY = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
)
MEMCPY_ASYNC = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_void_p,
)


class RuntimeAPI(ctypes.Structure):
    _fields_ = [
        ("get_device_count", GET_COUNT),
        ("set_device", SET_DEVICE),
        ("device_synchronize", SYNC),
        ("create_stream", CREATE_STREAM),
        ("destroy_stream", STREAM_OP),
        ("stream_synchronize", STREAM_OP),
        ("malloc_device", MALLOC),
        ("free_device", FREE),
        ("malloc_host", MALLOC),
        ("free_host", FREE),
        ("memcpy_sync", MEMCPY),
        ("memcpy_async", MEMCPY_ASYNC),
    ]


def test_runtime() -> None:
    lib = load_lib()
    lib.llaisysGetRuntimeAPI.argtypes = [ctypes.c_int]
    lib.llaisysGetRuntimeAPI.restype = ctypes.POINTER(RuntimeAPI)
    api_ptr = lib.llaisysGetRuntimeAPI(CPU)
    assert api_ptr and api_ptr.contents.get_device_count() == 1
    assert error(lib)[0] == 0
    api = api_ptr.contents
    api.set_device(0)
    assert error(lib)[0] == 0
    api.set_device(1)
    assert error(lib)[0] == INVALID_ARGUMENT

    for size in (0, 1, 4096, 1024 * 1024):
        src = (ctypes.c_ubyte * max(1, size))()
        dst = (ctypes.c_ubyte * max(1, size))()
        for i in range(size):
            src[i] = (i * 17 + 3) & 0xFF
        device_a = api.malloc_device(size)
        assert error(lib)[0] == 0
        device_b = api.malloc_device(size)
        assert error(lib)[0] == 0
        api.memcpy_sync(device_a, src, size, 1)
        assert error(lib)[0] == 0
        api.memcpy_sync(device_b, device_a, size, 3)
        assert error(lib)[0] == 0
        api.memcpy_sync(dst, device_b, size, 2)
        assert error(lib)[0] == 0
        assert bytes(src[:size]) == bytes(dst[:size])
        api.free_device(device_a)
        api.free_device(device_b)
        assert error(lib)[0] == 0

    for _ in range(256):
        ptr = api.malloc_device(4096)
        assert ptr and error(lib)[0] == 0
        api.free_device(ptr)
        assert error(lib)[0] == 0
    api.free_device(None)
    assert error(lib)[0] == 0

    unsupported = lib.llaisysGetRuntimeAPI(99)
    assert not unsupported
    assert error(lib)[0] in (INVALID_ARGUMENT, NOT_SUPPORTED)

    survivors = []

    def lifecycle_worker(worker_id):
        local = load_lib()
        for iteration in range(128):
            handle = make_tensor(local, (2, 3))
            payload = (ctypes.c_float * 6)(
                *[float(worker_id * 1000 + iteration + i) for i in range(6)]
            )
            local.tensorLoad(handle, payload)
            require_success(local)
            if iteration == 127:
                survivors.append((local, handle))
            else:
                local.tensorDestroy(handle)
                require_success(local)

    threads = [
        threading.Thread(target=lifecycle_worker, args=(worker_id,))
        for worker_id in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(survivors) == 8
    for local, handle in survivors:
        assert local.tensorGetNdim(handle) == 2
        require_success(local)
        local.tensorDestroy(handle)
        require_success(local)
    print("M2-T01 PASS")
    print("M2-T02 PASS")
    print("M2-T03 PASS")


if __name__ == "__main__":
    test_runtime()
