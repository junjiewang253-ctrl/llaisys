#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

// RoPE CPU 内核：对 in 做旋转位置编码写入 out
// - out/in: 连续(contiguous)，同 dtype，形状 [seqlen, nhead, d]（或 [seqlen, nkvhead, d]）
// - pos_ids: 连续，dtype=int64，形状 [seqlen]
// - dtype: out/in 的数据类型（F32/F16/BF16）
// - theta: 频率基数（常用 10000）
void rope(std::byte *out,
          const std::byte *in,
          const std::byte *pos_ids,
          llaisysDataType_t dtype,
          size_t seqlen,
          size_t nhead,
          size_t d,
          float theta);

} // namespace llaisys::ops::cpu