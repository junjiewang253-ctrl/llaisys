#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {

// Embedding（CPU 内核）
// out:   [N, D]   输出矩阵
// index: [N]      索引向量（必须 int64）
// weight:[V, D]   词表矩阵/权重矩阵（按行取 embedding）
void embedding(std::byte *out,
               const std::byte *index,
               const std::byte *weight,
               llaisysDataType_t out_type,
               llaisysDataType_t index_type,
               llaisysDataType_t weight_type,
               size_t N, // index 长度
               size_t D, // embedding 维度（每行长度）
               size_t V); // weight 的行数（词表大小）
} // namespace llaisys::ops::cpu