#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// CPU 后端 argmax 实现（纯计算内核）
// - max_idx：输出索引指针（int64，1 个元素）
// - max_val：输出最大值指针（与 vals 相同 dtype，1 个元素）
// - vals：输入数组指针（长度 numel）
// - type：vals/max_val 的 dtype（F32/F16/BF16）
// - numel：元素数量（不是字节数）
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel);
}