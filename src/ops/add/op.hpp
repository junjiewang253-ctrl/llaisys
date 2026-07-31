#pragma once

#include "../../tensor/tensor.hpp" // 引入tensor_t类型

namespace llaisys::ops { // llaisys::ops：算子统一的 C++ 命名空间
void add(tensor_t c, tensor_t a, tensor_t b); // add(tensor_t c, tensor_t a, tensor_t b)：算子入口。设计上是“把结果写入 c”，不是返回一个新张量（更接近底层框架/运行时风格，减少临时分配）
}
