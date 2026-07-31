import ctypes

from .. import LIB_LLAISYS as lib
from ..tensor import llaisysTensor_t  # 你们已有的 tensor 句柄类型

class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_int),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]

class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_o_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_gate_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_up_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_down_w", ctypes.POINTER(llaisysTensor_t)),
    ]

LlaisysQwen2Model_p = ctypes.c_void_p

lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model_p
lib.llaisysQwen2ModelCreate.argtypes = [
    ctypes.POINTER(LlaisysQwen2Meta),
    ctypes.c_int,                 # device
    ctypes.POINTER(ctypes.c_int), # device_ids
    ctypes.c_int,                 # ndevice
]

lib.llaisysQwen2ModelDestroy.restype = None
lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model_p]

lib.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)
lib.llaisysQwen2ModelWeights.argtypes = [LlaisysQwen2Model_p]

lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64
lib.llaisysQwen2ModelInfer.argtypes = [
    LlaisysQwen2Model_p,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_size_t,
]