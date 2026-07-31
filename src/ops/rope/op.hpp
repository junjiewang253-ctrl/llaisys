#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

// RoPE 算子入口：将结果写入 out（不返回新张量）
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
}
