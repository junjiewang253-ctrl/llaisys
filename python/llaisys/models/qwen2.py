# python/llaisys/models/qwen2.py
import ctypes
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
from safetensors import safe_open

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = e

from llaisys.libllaisys import LIB_LLAISYS as lib
from llaisys.libllaisys.models.qwen2 import (
    LlaisysQwen2Meta,
    LlaisysQwen2Weights,
    LlaisysQwen2Model_p,
)

# ---- enums: 跟 llaisys.h / llaisys_types.py 对齐 ----
LLAISYS_DEVICE_CPU = 0

LLAISYS_DTYPE_I64 = 6
LLAISYS_DTYPE_F16 = 12
LLAISYS_DTYPE_F32 = 13
LLAISYS_DTYPE_BF16 = 19


def _parse_dtype(dtype: str) -> int:
    d = (dtype or "").lower().strip()
    if d in ("auto",):
        return -1
    if d in ("bf16", "bfloat16"):
        return LLAISYS_DTYPE_BF16
    if d in ("f16", "fp16", "float16"):
        return LLAISYS_DTYPE_F16
    if d in ("f32", "fp32", "float32"):
        return LLAISYS_DTYPE_F32
    raise ValueError(f"Unsupported dtype='{dtype}', expected one of: auto/bf16/f16/f32")


def _ensure_torch():
    if torch is None:
        raise RuntimeError(
            "This loader requires PyTorch to handle bfloat16 safetensors.\n"
            f"Original import error: {_TORCH_IMPORT_ERROR!r}\n"
            "Fix: pip install torch"
        )


def _set_ctypes_signatures():
    # 只设置 qwen2 model 相关签名；tensor/ops 在 llaisys.libllaisys.__init__ 里已经 load_* 过了
    lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model_p
    lib.llaisysQwen2ModelCreate.argtypes = [
        ctypes.POINTER(LlaisysQwen2Meta),
        ctypes.c_int,  # device
        ctypes.POINTER(ctypes.c_int),  # device_ids
        ctypes.c_int,  # ndevice
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


def _tensor_from_numpy(
    arr: np.ndarray,
    *,
    llaisys_dtype: int,
    device: int = LLAISYS_DEVICE_CPU,
    device_id: int = 0,
) -> ctypes.c_void_p:
    # 注意：这里依赖 libllaisys.__init__ 里 load_tensor(LIB_LLAISYS) 已经设好 tensorCreate/tensorLoad 签名
    arr = np.ascontiguousarray(arr)
    shape = (ctypes.c_size_t * arr.ndim)(*arr.shape)
    t = lib.tensorCreate(shape, arr.ndim, int(llaisys_dtype), int(device), int(device_id))
    lib.tensorLoad(t, ctypes.c_void_p(arr.ctypes.data))
    return t


def _build_weight_map(model_dir: str) -> Optional[Dict[str, str]]:
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(idx):
        return None
    with open(idx, "r", encoding="utf-8") as f:
        j = json.load(f)
    wm = j.get("weight_map", {})
    return {k: os.path.join(model_dir, v) for k, v in wm.items()}


def _iter_safetensors_files(model_dir: str):
    for fn in sorted(os.listdir(model_dir)):
        if fn.endswith(".safetensors"):
            yield os.path.join(model_dir, fn)


def _load_torch_tensor_by_key(model_dir: str, key: str, weight_map: Optional[Dict[str, str]]):
    _ensure_torch()

    if weight_map is not None and key in weight_map:
        path = weight_map[key]
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    for path in _iter_safetensors_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)

    raise KeyError(key)


def _torch_to_numpy_for_llaisys(t: "torch.Tensor", target_llaisys_dtype: int) -> Tuple[np.ndarray, int]:
    _ensure_torch()
    t = t.detach().cpu().contiguous()

    if t.dtype == torch.int64:
        return t.numpy(), LLAISYS_DTYPE_I64

    if target_llaisys_dtype == LLAISYS_DTYPE_BF16:
        if t.dtype != torch.bfloat16:
            t = t.to(torch.bfloat16)
        # bit-cast: bf16 -> uint16 view
        return t.view(torch.uint16).numpy(), LLAISYS_DTYPE_BF16

    if target_llaisys_dtype == LLAISYS_DTYPE_F16:
        if t.dtype != torch.float16:
            t = t.to(torch.float16)
        return t.numpy(), LLAISYS_DTYPE_F16

    if target_llaisys_dtype == LLAISYS_DTYPE_F32:
        if t.dtype != torch.float32:
            t = t.to(torch.float32)
        return t.numpy(), LLAISYS_DTYPE_F32

    raise ValueError(f"Unsupported target llaisys dtype: {target_llaisys_dtype}")


@dataclass
class Qwen2Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    eos_token_id: int


def _read_config(model_dir: str) -> Qwen2Config:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return Qwen2Config(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg.get("max_position_embeddings", cfg.get("max_sequence_length", 4096)),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        rope_theta=float(cfg.get("rope_theta", 10000.0)),
        eos_token_id=int(cfg.get("eos_token_id", cfg.get("eos_token_ids", [0])[0])),
    )


def _infer_model_dtype_from_weights(model_dir: str, weight_map: Optional[Dict[str, str]]) -> int:
    _ensure_torch()
    t = _load_torch_tensor_by_key(model_dir, "model.embed_tokens.weight", weight_map)
    if t.dtype == torch.bfloat16:
        return LLAISYS_DTYPE_BF16
    if t.dtype == torch.float16:
        return LLAISYS_DTYPE_F16
    return LLAISYS_DTYPE_F32


def _normalize_input_ids(input_ids: Any) -> list[int]:
    """
    兼容 HF: input_ids 可能是
    - list[int]
    - np.ndarray shape (seq,) or (1, seq)
    - torch.Tensor shape (seq,) or (1, seq)
    - dict 里带 "input_ids"
    """
    if isinstance(input_ids, dict) and "input_ids" in input_ids:
        input_ids = input_ids["input_ids"]

    if torch is not None and isinstance(input_ids, torch.Tensor):
        x = input_ids.detach().cpu()
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError(f"Only batch=1 is supported, got shape={tuple(x.shape)}")
            x = x[0]
        return [int(v) for v in x.tolist()]

    if isinstance(input_ids, np.ndarray):
        x = input_ids
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError(f"Only batch=1 is supported, got shape={tuple(x.shape)}")
            x = x[0]
        return [int(v) for v in x.tolist()]

    if isinstance(input_ids, (list, tuple)):
        return [int(v) for v in input_ids]

    raise TypeError(f"Unsupported input_ids type: {type(input_ids)}")


class Qwen2:
    def __init__(self, model_dir: str, device: str = "cpu", dtype: str = "f16"):
        _set_ctypes_signatures()

        self.model_dir = model_dir
        self.device = LLAISYS_DEVICE_CPU
        self.device_id = 0

        self._weight_map = _build_weight_map(model_dir)

        wanted = _parse_dtype(dtype)
        if wanted == -1:  # auto
            wanted = _infer_model_dtype_from_weights(model_dir, self._weight_map)
        self._wanted_dtype = wanted

        cfg = _read_config(model_dir)

        meta = LlaisysQwen2Meta()
        meta.dtype = int(self._wanted_dtype)
        meta.nlayer = cfg.num_hidden_layers
        meta.hs = cfg.hidden_size
        meta.nh = cfg.num_attention_heads
        meta.nkvh = cfg.num_key_value_heads
        meta.dh = cfg.hidden_size // cfg.num_attention_heads
        meta.di = cfg.intermediate_size
        meta.maxseq = cfg.max_position_embeddings
        meta.voc = cfg.vocab_size
        meta.epsilon = float(cfg.rms_norm_eps)
        meta.theta = float(cfg.rope_theta)
        meta.end_token = int(cfg.eos_token_id)

        self.meta = meta
        self._model: LlaisysQwen2Model_p = lib.llaisysQwen2ModelCreate(ctypes.byref(meta), self.device, None, 0)
        if not self._model:
            raise RuntimeError("llaisysQwen2ModelCreate failed")

        self._w_ptr = lib.llaisysQwen2ModelWeights(self._model)
        if not self._w_ptr:
            raise RuntimeError("llaisysQwen2ModelWeights failed")

        self._load_weights()

    def __del__(self):
        try:
            if getattr(self, "_model", None):
                lib.llaisysQwen2ModelDestroy(self._model)
                self._model = None
        except Exception:
            pass

    def _load_weight_tensor(self, key: str) -> ctypes.c_void_p:
        tt = _load_torch_tensor_by_key(self.model_dir, key, self._weight_map)
        arr, dt = _torch_to_numpy_for_llaisys(tt, self._wanted_dtype)
        return _tensor_from_numpy(arr, llaisys_dtype=dt, device=self.device, device_id=self.device_id)

    def _load_weights(self):
        w = self._w_ptr.contents
        L = int(self.meta.nlayer)

        w.in_embed = self._load_weight_tensor("model.embed_tokens.weight")

        try:
            w.out_embed = self._load_weight_tensor("lm_head.weight")
        except KeyError:
            w.out_embed = w.in_embed  # tied

        w.out_norm_w = self._load_weight_tensor("model.norm.weight")

        for i in range(L):
            pref = f"model.layers.{i}."

            w.attn_norm_w[i] = self._load_weight_tensor(pref + "input_layernorm.weight")

            w.attn_q_w[i] = self._load_weight_tensor(pref + "self_attn.q_proj.weight")
            w.attn_k_w[i] = self._load_weight_tensor(pref + "self_attn.k_proj.weight")
            w.attn_v_w[i] = self._load_weight_tensor(pref + "self_attn.v_proj.weight")
            w.attn_o_w[i] = self._load_weight_tensor(pref + "self_attn.o_proj.weight")

            for name, arrptr in [
                ("self_attn.q_proj.bias", w.attn_q_b),
                ("self_attn.k_proj.bias", w.attn_k_b),
                ("self_attn.v_proj.bias", w.attn_v_b),
            ]:
                try:
                    arrptr[i] = self._load_weight_tensor(pref + name)
                except KeyError:
                    arrptr[i] = None

            w.mlp_norm_w[i] = self._load_weight_tensor(pref + "post_attention_layernorm.weight")
            w.mlp_gate_w[i] = self._load_weight_tensor(pref + "mlp.gate_proj.weight")
            w.mlp_up_w[i] = self._load_weight_tensor(pref + "mlp.up_proj.weight")
            w.mlp_down_w[i] = self._load_weight_tensor(pref + "mlp.down_proj.weight")

    def infer_next(self, token_ids: list[int]) -> int:
        arr = (ctypes.c_int64 * len(token_ids))(*token_ids)
        out = int(lib.llaisysQwen2ModelInfer(self._model, arr, len(token_ids)))
        if out < 0:
            # 对齐 qwen2.cc 的错误码：
            # -1 bad args, -2 weights not ready, -3 too long, -4 no new token, -100 data null
            raise RuntimeError(f"llaisysQwen2ModelInfer failed: code={out}, ntoken={len(token_ids)}")
        return out

    def generate(self, input_ids=None, **kwargs):
        """
        兼容 test/test_infer.py 常见调用方式（类 HuggingFace）：
          model.generate(input_ids, max_new_tokens=..., eos_token_id=..., ...)
        仅支持 batch=1。
        """
        tokens = _normalize_input_ids(input_ids)

        eos_token_id = kwargs.get("eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = int(self.meta.end_token)

        max_new_tokens = kwargs.get("max_new_tokens", None)
        max_length = kwargs.get("max_length", None)

        if max_new_tokens is None:
            if max_length is not None:
                max_new_tokens = max(0, int(max_length) - len(tokens))
            else:
                max_new_tokens = 128  # 默认给个值，避免死循环
        max_new_tokens = int(max_new_tokens)

        # 关键：不要超过 C++ 侧 meta.maxseq，否则会返回 -3
        maxseq = int(self.meta.maxseq)

        for _ in range(max_new_tokens):
            if len(tokens) >= maxseq:
                break  # 这句就放在这里（每次推理前做上限检查）

            nxt = self.infer_next(tokens)
            tokens.append(int(nxt))

            if int(nxt) == int(eos_token_id):
                break

        return tokens