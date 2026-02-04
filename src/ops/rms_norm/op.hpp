#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// rms_norm：RMS Normalization（把结果写入 out）
// 约定（当前版本）：
// - in/out 为 2D contiguous 张量 [M, D]
// - weight 为 1D contiguous 张量 [D]
// - 不做 broadcasting
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
}
