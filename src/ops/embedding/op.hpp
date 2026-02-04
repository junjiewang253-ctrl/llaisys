#pragma once

#include "../../tensor/tensor.hpp" // 引入tensor_t

namespace llaisys::ops {
	// embedding算子入口：把结果写入out(不返回新张量)
void embedding(tensor_t out, tensor_t index, tensor_t weight);
} // namespace llaisys::ops
