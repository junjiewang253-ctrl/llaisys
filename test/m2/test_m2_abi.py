#!/usr/bin/env python3
import ctypes
import os
import threading
from pathlib import Path


SUCCESS = 0
INVALID_ARGUMENT = 1
OUT_OF_MEMORY = 2
NOT_SUPPORTED = 3
INTERNAL_ERROR = 4
CPU = 0
NVIDIA = 1
F32 = 13
INVALID_DTYPE = 0


def library_path() -> Path:
    explicit = os.environ.get("LLAISYS_TEST_LIB")
    if explicit:
        return Path(explicit).resolve()
    import llaisys
    from llaisys.libllaisys import LIB_LLAISYS

    return Path(LIB_LLAISYS._name).resolve()


def load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(library_path()))
    lib.llaisysGetLastErrorCode.argtypes = []
    lib.llaisysGetLastErrorCode.restype = ctypes.c_int
    lib.llaisysGetLastErrorMessage.argtypes = []
    lib.llaisysGetLastErrorMessage.restype = ctypes.c_char_p
    lib.llaisysClearLastError.argtypes = []
    lib.llaisysClearLastError.restype = None

    lib.tensorCreate.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.tensorCreate.restype = ctypes.c_void_p
    lib.tensorDestroy.argtypes = [ctypes.c_void_p]
    lib.tensorDestroy.restype = None
    lib.tensorGetData.argtypes = [ctypes.c_void_p]
    lib.tensorGetData.restype = ctypes.c_void_p
    lib.tensorGetNdim.argtypes = [ctypes.c_void_p]
    lib.tensorGetNdim.restype = ctypes.c_size_t
    lib.tensorGetShape.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.tensorGetShape.restype = None
    lib.tensorGetStrides.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ssize_t),
    ]
    lib.tensorGetStrides.restype = None
    lib.tensorGetDataType.argtypes = [ctypes.c_void_p]
    lib.tensorGetDataType.restype = ctypes.c_int
    lib.tensorGetDeviceType.argtypes = [ctypes.c_void_p]
    lib.tensorGetDeviceType.restype = ctypes.c_int
    lib.tensorGetDeviceId.argtypes = [ctypes.c_void_p]
    lib.tensorGetDeviceId.restype = ctypes.c_int
    lib.tensorIsContiguous.argtypes = [ctypes.c_void_p]
    lib.tensorIsContiguous.restype = ctypes.c_uint8
    lib.tensorLoad.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.tensorLoad.restype = None
    lib.tensorView.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    lib.tensorView.restype = ctypes.c_void_p
    lib.tensorPermute.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.tensorPermute.restype = ctypes.c_void_p
    lib.tensorSlice.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.tensorSlice.restype = ctypes.c_void_p
    lib.tensorContiguous.argtypes = [ctypes.c_void_p]
    lib.tensorContiguous.restype = ctypes.c_void_p
    lib.tensorReshape.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    lib.tensorReshape.restype = ctypes.c_void_p
    lib.tensorTo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.tensorTo.restype = ctypes.c_void_p
    return lib


def error(lib: ctypes.CDLL) -> tuple[int, str]:
    code = int(lib.llaisysGetLastErrorCode())
    raw = lib.llaisysGetLastErrorMessage()
    return code, raw.decode("utf-8", errors="replace") if raw else ""


def require_success(lib: ctypes.CDLL) -> None:
    code, message = error(lib)
    if code != SUCCESS:
        raise AssertionError(f"unexpected C error {code}: {message}")


def make_tensor(lib: ctypes.CDLL, shape, dtype=F32):
    shape_buf = (
        None
        if shape is None
        else (ctypes.c_size_t * len(shape))(*shape)
    )
    ndim = 0 if shape is None else len(shape)
    handle = lib.tensorCreate(shape_buf, ndim, dtype, CPU, 0)
    require_success(lib)
    assert handle
    return handle


def test_error_fence() -> None:
    lib = load_lib()
    lib.llaisysClearLastError()
    assert error(lib) == (SUCCESS, "")

    assert not lib.tensorCreate(None, 1, F32, CPU, 0)
    code, message = error(lib)
    assert code == INVALID_ARGUMENT and message

    scalar = make_tensor(lib, None)
    assert lib.tensorGetNdim(scalar) == 0
    require_success(lib)
    lib.tensorDestroy(scalar)
    require_success(lib)
    lib.tensorDestroy(None)
    require_success(lib)

    assert not lib.tensorCreate(None, 0, INVALID_DTYPE, CPU, 0)
    assert error(lib)[0] == INVALID_ARGUMENT
    assert not lib.tensorCreate(None, 0, F32, NVIDIA, 0)
    assert error(lib)[0] == NOT_SUPPORTED

    assert lib.tensorGetNdim(None) == ctypes.c_size_t(-1).value
    assert error(lib)[0] == INVALID_ARGUMENT
    assert lib.tensorGetDataType(None) == INVALID_DTYPE
    assert error(lib)[0] == INVALID_ARGUMENT

    results = []

    def worker(dtype):
        local = load_lib()
        local.tensorCreate(None, 0, dtype, CPU, 0)
        results.append(error(local))

    threads = [
        threading.Thread(target=worker, args=(INVALID_DTYPE,)),
        threading.Thread(target=worker, args=(999,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 2
    assert all(code == INVALID_ARGUMENT and message for code, message in results)
    assert error(lib)[0] == INVALID_ARGUMENT

    # The public Python wrapper must map C status deterministically.
    import llaisys

    try:
        llaisys.Tensor((2, -1))
    except ValueError:
        pass
    else:
        raise AssertionError("negative Python shape was not rejected")

    print("M2-T18 PASS")
    print("M2-T19 PASS")


if __name__ == "__main__":
    test_error_fence()
