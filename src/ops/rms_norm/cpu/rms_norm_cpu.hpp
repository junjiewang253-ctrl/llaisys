#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {

// RMSNorm（CPU 后端）
// - out/in：视为二维张量 [M, D]，且必须 contiguous、row-major
// - weight：一维张量 [D]，且必须 contiguous
// - eps：防止除零的小数
void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              llaisysDataType_t type,
              size_t M, size_t D, float eps);
} // namespace llaisys::ops::cpu