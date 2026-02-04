#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {

// 线性层（CPU 内核）
// out:    [M, N] 输出
// in:     [M, K] 输入
// weight: [N, K] 权重（注意：计算用的是 W^T，因此访问方式是 weight[n, k]）
// bias:   [N] 或 nullptr（可选）
// dtype:  out/in/weight（以及 bias 若存在）必须一致，只支持 F32/F16/BF16
void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias, // 可为空
            llaisysDataType_t dtype,
            size_t M, size_t K, size_t N,
            bool has_bias);
} // namespace llaisys::ops::cpu
