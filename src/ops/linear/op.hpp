#pragma once

#include "../../tensor/tensor.hpp" // 引入 tensor_t

namespace llaisys::ops {

// Linear 算子入口：把结果写入 out
// bias 可选：允许传 nullptr（表示没有 bias）
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
} // namespace llaisys::ops
