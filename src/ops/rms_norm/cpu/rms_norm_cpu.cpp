#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <type_traits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#undef __C
#include <immintrin.h>

namespace {

// 将任意支持类型转成 float 参与计算：
// - 对 BF16/F16：先 cast 到 float
// - 对 F32：直接 static_cast
template <typename T>
inline float to_f32(T v) {
	if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
		return llaisys::utils::cast<float>(v);
    } else {
        return static_cast<float>(v);
	}
}

// 将 float 结果转回目标类型 T：
// - 对 BF16/F16：cast 回去
// - 对 F32：直接 static_cast
template <typename T>
inline T from_f32(float v) {
    if constexpr (std::is_same_v <T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::cast<T>(v);
    } else {
        return static_cast<T>(v);
    }
}

__attribute__((target("avx2,fma")))
__m256 load_bf16_square_avx2(
    const llaisys::bf16_t *values, size_t vector_index) {
    constexpr size_t vector_width = 8;
    const __m128i packed = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(
            values + vector_index * vector_width));
    const __m256 floats = _mm256_castsi256_ps(
        _mm256_slli_epi32(
            _mm256_cvtepu16_epi32(packed), 16));
    return _mm256_mul_ps(floats, floats);
}

__attribute__((target("avx2,fma")))
float bf16_sum_squares_avx2(
    const llaisys::bf16_t *values, size_t count) {
    constexpr size_t vector_width = 8;
    constexpr size_t ilp_factor = 4;
    constexpr size_t levels_count = 4;
    __m256 levels[levels_count][ilp_factor];
    for (size_t level = 0; level < levels_count; ++level) {
        for (size_t partial = 0; partial < ilp_factor; ++partial) {
            levels[level][partial] = _mm256_setzero_ps();
        }
    }

    const size_t vector_count = count / vector_width;
    const size_t group_count = vector_count / ilp_factor;
    size_t ceil_log2 = 0;
    for (size_t value = group_count > 0 ? group_count - 1 : 0;
         value > 0; value >>= 1) {
        ++ceil_log2;
    }
    const size_t level_power = std::max<size_t>(
        4, ceil_log2 / levels_count);
    const size_t level_step = size_t{1} << level_power;
    const size_t level_mask = level_step - 1;
    size_t group = 0;
    while (group + level_step <= group_count) {
        for (size_t item = 0; item < level_step; ++item, ++group) {
            for (size_t partial = 0; partial < ilp_factor; ++partial) {
                levels[0][partial] = _mm256_add_ps(
                    levels[0][partial],
                    load_bf16_square_avx2(
                        values, group * ilp_factor + partial));
            }
        }
        for (size_t level = 1; level < levels_count; ++level) {
            for (size_t partial = 0; partial < ilp_factor; ++partial) {
                levels[level][partial] = _mm256_add_ps(
                    levels[level][partial],
                    levels[level - 1][partial]);
                levels[level - 1][partial] = _mm256_setzero_ps();
            }
            const size_t mask = level_mask << (level * level_power);
            if ((group & mask) != 0) {
                break;
            }
        }
    }
    while (group < group_count) {
        for (size_t partial = 0; partial < ilp_factor; ++partial) {
            levels[0][partial] = _mm256_add_ps(
                levels[0][partial],
                load_bf16_square_avx2(
                    values, group * ilp_factor + partial));
        }
        ++group;
    }
    for (size_t level = 1; level < levels_count; ++level) {
        for (size_t partial = 0; partial < ilp_factor; ++partial) {
            levels[0][partial] = _mm256_add_ps(
                levels[0][partial], levels[level][partial]);
        }
    }
    for (size_t vector_index = group_count * ilp_factor;
         vector_index < vector_count; ++vector_index) {
        levels[0][0] = _mm256_add_ps(
            levels[0][0],
            load_bf16_square_avx2(values, vector_index));
    }
    __m256 accumulator = _mm256_add_ps(
        levels[0][0], levels[0][1]);
    accumulator = _mm256_add_ps(accumulator, levels[0][2]);
    accumulator = _mm256_add_ps(accumulator, levels[0][3]);

    size_t index = vector_count * vector_width;
    float result = 0.0f;
    for (; index < count; ++index) {
        const float value = to_f32(values[index]);
        result += value * value;
    }
    alignas(32) float lanes[vector_width];
    _mm256_store_ps(lanes, accumulator);
    for (size_t lane = 0; lane < vector_width; ++lane) {
        result += lanes[lane];
    }
    return result;
}

// RMSNorm 核心计算（朴素实现，按行做归一化）
// 输入输出视为：
// - in/out: [M, D]
// - weight: [D]
//
// 对每一行 m：
// 1) mean_sq = (1/D) * sum_i in[m,i]^2
// 2) inv_rms = 1 / sqrt(mean_sq + eps)
// 3) out[m,i] = in[m,i] * inv_rms * weight[i]
template <typename T>
void rms_norm_(T* out, const T* in, const T* weight, size_t M, size_t D, float eps) {
    for (size_t m = 0; m < M; ++m) {
        // 指向第m行起始位置
        const T *in_row = in + m * D; 
        T *out_row = out + m * D;

        // 第一步：计算均方mean(x^2>
        float sum_sq = 0.0f;
        if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
            // PyTorch 2.4.0 dispatches the fixed CPU sum kernel used by
            // Qwen2 RMSNorm through its AVX2 cascade path on this supported
            // wheel, even when the host also reports AVX-512 capability.
            if (__builtin_cpu_supports("avx2") && D >= 8) {
                sum_sq = bf16_sum_squares_avx2(in_row, D);
            } else {
                for (size_t i = 0; i < D; ++i) {
                    const float x = to_f32(in_row[i]);
                    sum_sq += x * x;
                }
            }
        } else {
            for (size_t i = 0; i < D; ++i) {
                const float x = to_f32(in_row[i]);
                sum_sq += x * x;
            }
        }
        float mean_sq = sum_sq / static_cast<float>(D);

        // 第二步：计算缩放系数 inv_rms
        // 注意 eps 用来避免 mean_sq 为 0 时除零
        float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

        // 第三步：逐元素归一化并乘以 weight（仿射缩放）
        for (size_t i = 0; i < D; ++i) {
            float x = to_f32(in_row[i]); 
            float w = to_f32(weight[i]);
            float normalized = x * inv_rms;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                          std::is_same_v<T, llaisys::fp16_t>) {
                // Qwen2/PyTorch casts the normalized activation back to the
                // input dtype before applying the learned scale.
                normalized = to_f32(from_f32<T>(normalized));
            }
            float y = normalized * w;
            out_row[i] = from_f32<T>(y);
        }
    }
}
} // namespace

namespace llaisys::ops::cpu {
    void rms_norm(std::byte* out,
    const std::byte* in,
    const std::byte* weight,
    llaisysDataType_t type,
    size_t M, size_t D, float eps) {
    // 用 dtype 枚举进行运行时分发：把 byte 指针转成具体类型指针，然后调用模板内核
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight),
                         M, D, eps);

    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight),
                         M, D, eps);

    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight),
                         M, D, eps);
    default:
        // 其他 dtype（如 int8/int64）未实现
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
