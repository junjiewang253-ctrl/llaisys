from typing import Sequence, Tuple

from .libllaisys import (
    LIB_LLAISYS,
    llaisysTensor_t,
    llaisysDeviceType_t,
    DeviceType,
    llaisysDataType_t,
    DataType,
    check_last_error,
)
from ctypes import c_size_t, c_int, c_ssize_t, c_void_p


class Tensor:
    def __init__(
        self,
        shape: Sequence[int] = None,
        dtype: DataType = DataType.F32,
        device: DeviceType = DeviceType.CPU,
        device_id: int = 0,
        tensor: llaisysTensor_t = None,
    ):
        if tensor:
            self._tensor = tensor
        else:
            if shape is not None:
                if any(not isinstance(dim, int) for dim in shape):
                    raise TypeError("tensor shape dimensions must be integers")
                if any(dim < 0 for dim in shape):
                    raise ValueError("tensor shape dimensions must be non-negative")
            _ndim = 0 if shape is None else len(shape)
            _shape = None if shape is None else (c_size_t * len(shape))(*shape)
            self._tensor: llaisysTensor_t = LIB_LLAISYS.tensorCreate(
                _shape,
                c_size_t(_ndim),
                llaisysDataType_t(dtype),
                llaisysDeviceType_t(device),
                c_int(device_id),
            )
            check_last_error()

    def __del__(self):
        if hasattr(self, "_tensor") and self._tensor is not None:
            LIB_LLAISYS.tensorDestroy(self._tensor)
            self._tensor = None

    def shape(self) -> Tuple[int]:
        buf = (c_size_t * self.ndim())()
        LIB_LLAISYS.tensorGetShape(self._tensor, buf)
        check_last_error()
        return tuple(buf[i] for i in range(self.ndim()))

    def strides(self) -> Tuple[int]:
        buf = (c_ssize_t * self.ndim())()
        LIB_LLAISYS.tensorGetStrides(self._tensor, buf)
        check_last_error()
        return tuple(buf[i] for i in range(self.ndim()))

    def ndim(self) -> int:
        result = int(LIB_LLAISYS.tensorGetNdim(self._tensor))
        check_last_error()
        return result

    def dtype(self) -> DataType:
        result = LIB_LLAISYS.tensorGetDataType(self._tensor)
        check_last_error()
        return DataType(result)

    def device_type(self) -> DeviceType:
        result = LIB_LLAISYS.tensorGetDeviceType(self._tensor)
        check_last_error()
        return DeviceType(result)

    def device_id(self) -> int:
        result = int(LIB_LLAISYS.tensorGetDeviceId(self._tensor))
        check_last_error()
        return result

    def data_ptr(self) -> c_void_p:
        result = LIB_LLAISYS.tensorGetData(self._tensor)
        check_last_error()
        return result

    def lib_tensor(self) -> llaisysTensor_t:
        return self._tensor

    def debug(self):
        LIB_LLAISYS.tensorDebug(self._tensor)
        check_last_error()

    def __repr__(self):
        return f"<Tensor shape={self.shape}, dtype={self.dtype}, device={self.device_type}:{self.device_id}>"

    def load(self, data: c_void_p):
        LIB_LLAISYS.tensorLoad(self._tensor, data)
        check_last_error()

    def is_contiguous(self) -> bool:
        result = bool(LIB_LLAISYS.tensorIsContiguous(self._tensor))
        check_last_error()
        return result

    def view(self, *shape: int) -> llaisysTensor_t:
        if any(not isinstance(dim, int) for dim in shape):
            raise TypeError("view dimensions must be integers")
        if any(dim < 0 for dim in shape):
            raise ValueError("view dimensions must be non-negative")
        _shape = (c_size_t * len(shape))(*shape)
        result = LIB_LLAISYS.tensorView(
            self._tensor, _shape, c_size_t(len(shape))
        )
        check_last_error()
        return Tensor(tensor=result)

    def permute(self, *perm: int) -> llaisysTensor_t:
        if any(not isinstance(dim, int) for dim in perm):
            raise TypeError("permutation dimensions must be integers")
        if len(perm) != self.ndim() or any(dim < 0 for dim in perm):
            raise ValueError("invalid permutation")
        _perm = (c_size_t * len(perm))(*perm)
        result = LIB_LLAISYS.tensorPermute(self._tensor, _perm)
        check_last_error()
        return Tensor(tensor=result)

    def slice(self, dim: int, start: int, end: int):
        if any(not isinstance(value, int) for value in (dim, start, end)):
            raise TypeError("slice arguments must be integers")
        if dim < 0 or start < 0 or end < 0:
            raise ValueError("slice arguments must be non-negative")
        result = LIB_LLAISYS.tensorSlice(
            self._tensor, c_size_t(dim), c_size_t(start), c_size_t(end)
        )
        check_last_error()
        return Tensor(tensor=result)

    def contiguous(self):
        result = LIB_LLAISYS.tensorContiguous(self._tensor)
        check_last_error()
        return Tensor(tensor=result)

    def reshape(self, *shape: int):
        if any(not isinstance(dim, int) for dim in shape):
            raise TypeError("reshape dimensions must be integers")
        if any(dim < 0 for dim in shape):
            raise ValueError("reshape dimensions must be non-negative")
        shape_buffer = (c_size_t * len(shape))(*shape)
        result = LIB_LLAISYS.tensorReshape(
            self._tensor, shape_buffer, c_size_t(len(shape))
        )
        check_last_error()
        return Tensor(tensor=result)

    def to(self, device: DeviceType, device_id: int = -1):
        result = LIB_LLAISYS.tensorTo(
            self._tensor,
            llaisysDeviceType_t(device),
            c_int(device_id),
        )
        check_last_error()
        return Tensor(tensor=result)
