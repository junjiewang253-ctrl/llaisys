#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include<type_traits>

// 将不同类型的数转换为 float 用于比较（尤其是半精度类型）
// - 对 bf16/fp16：用项目提供的 cast 转为 float
// - 对 float：直接 static_cast
template <typename T>
static inline float to_float(T x) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::cast<float>(x);
    } else {
        return static_cast<float>(x);
    }
}

// 模板内核：在类型 T 已知时执行 argmax
// out_idx：输出最大值索引（int64）
// out_val：输出最大值（dtype 为 T）
// vals：输入数组（dtype 为 T）
// numel：元素个数（>0）
template <typename T>
static void argmax_(int64_t* out_idx, T* out_val, const T* vals, size_t numel) {
    // 调用者已确保numel>0
    size_t best_i = 0;
    float best_v = to_float(vals[0]);
    // 从第二个元素开始遍历，找到最大值及其索引
    for (size_t i = 1; i < numel; ++i) {
        float v = to_float(vals[i]);
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    *out_idx = static_cast<int64_t>(best_i);
    *out_val = vals[best_i]; // max_val保存为原dtype(T)
}

namespace llaisys::ops::cpu {
   
    void argmax(std::byte* max_idx, std::byte* max_val, const std::byte* vals, llaisysDataType_t type, size_t numel) {
        // max_idx强制按int64写入
        auto *out_i64 = reinterpret_cast<int64_t *>(max_idx);
        // 根据dtype分发到不同模板实例
        switch (type) {
        case LLAISYS_DTYPE_F32:
            return argmax_(out_i64, reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), numel);
        case LLAISYS_DTYPE_F16:
            return argmax_(out_i64, reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
        case LLAISYS_DTYPE_BF16:
            return argmax_(out_i64, reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}