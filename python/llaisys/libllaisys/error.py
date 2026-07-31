import ctypes
from enum import IntEnum


class Status(IntEnum):
    SUCCESS = 0
    INVALID_ARGUMENT = 1
    OUT_OF_MEMORY = 2
    NOT_SUPPORTED = 3
    INTERNAL_ERROR = 4


_library = None


def load_error(lib):
    global _library
    _library = lib
    lib.llaisysGetLastErrorCode.argtypes = []
    lib.llaisysGetLastErrorCode.restype = ctypes.c_int
    lib.llaisysGetLastErrorMessage.argtypes = []
    lib.llaisysGetLastErrorMessage.restype = ctypes.c_char_p
    lib.llaisysClearLastError.argtypes = []
    lib.llaisysClearLastError.restype = None


def check_last_error():
    if _library is None:
        raise RuntimeError("LLAISYS error API is not initialized")
    code = Status(_library.llaisysGetLastErrorCode())
    if code == Status.SUCCESS:
        return
    raw = _library.llaisysGetLastErrorMessage()
    message = raw.decode("utf-8", errors="replace") if raw else code.name
    if code == Status.INVALID_ARGUMENT:
        raise ValueError(message)
    if code == Status.OUT_OF_MEMORY:
        raise MemoryError(message)
    if code == Status.NOT_SUPPORTED:
        raise NotImplementedError(message)
    raise RuntimeError(message)
