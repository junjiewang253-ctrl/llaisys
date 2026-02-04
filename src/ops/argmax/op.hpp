#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// Argmax 算子入口（框架层）
// - vals：输入张量，暂时假设为 1D 连续张量 [N]
// - max_val：输出最大值张量，暂时假设为 1D 且只有 1 个元素 [1]，dtype 与 vals 相同
// - max_idx：输出最大值索引张量，暂时假设为 1D 且只有 1 个元素 [1]，dtype 必须是 int64
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
}
