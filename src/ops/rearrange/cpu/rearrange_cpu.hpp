#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

// 将 in 按 stride 读，写入 out（也按 out stride 写）
// shape/ndim/stride 由上层 (op.cpp) 负责提供并检查一致性。
void rearrange(
    std::byte *out_data,
    const std::byte *in_data,
    llaisysDataType_t dtype,
    const size_t *shape,
    const ptrdiff_t *out_strides, // 单位：元素 (elements)，不是 bytes
    const ptrdiff_t *in_strides,  // 单位：元素 (elements)
    size_t ndim);

} // namespace llaisys::ops::cpu