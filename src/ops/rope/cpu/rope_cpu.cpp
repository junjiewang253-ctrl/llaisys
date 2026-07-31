#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>
#include <type_traits>

// 将任意支持的标量类型转成float参与计算（半精度需要转float）
template <typename T>
static inline float to_f32(T x) {
    return llaisys::utils::cast<float>(x);
}
template <>
inline float to_f32<float>(float x) {
    return x;
}

// 将 float 结果转回目标类型（半精度需要 cast 回去）
template <typename T>
static inline T from_f32(float x) {
    return llaisys::utils::cast<T>(x);
}
template <>
inline float from_f32<float>(float x) {
    return x;
}

// 模板内核：对指定 dtype 执行 RoPE
// 数据布局假设：in/out 是连续的 [seqlen, nhead, d]
template <typename T>
void rope_(T* out, const T* in, const int64_t* pos_ids,
    size_t seqlen, size_t nhead, size_t d, float theta) {
    // d必须为偶数，前半段是a，后半段是b
    const size_t half = d / 2;

    // 为了高效计算 theta^(-2j/d)，用 exp(-log(theta) * 2j/d)
    // 用 double 做频率/角度计算，减少相位误差（特别是 head_dim 很大时）
    const double log_theta = std::log(static_cast<double>(theta));

    // 外层遍历token位置s(对应pos_ids[s])
    for (size_t s = 0; s < seqlen; ++s) {
        // p 是位置
        const double p = static_cast<double>(pos_ids[s]);

        // 遍历head维度
        for (size_t h = 0; h < nhead; ++h) {
            // 计算该向量在扁平数组中的起始偏移
            // index = ((s * nhead + h) * d + k)
            const size_t base = (s * nhead + h) * d;

            // 对每个 j（0..d/2-1）计算旋转
            for (size_t j = 0; j < half; ++j) {
                // inv_freq_j = theta^(-2j/d)
                const double inv_freq = std::exp(-log_theta * (2.0 * static_cast<double>(j) / static_cast<double>(d)));

                // phi_{s,j} = p_s * inv_freq_j
                const double phi = p * inv_freq;

                // 预计算 cos/sin
                const float c = static_cast<float>(std::cos(phi));
                const float si = static_cast<float>(std::sin(phi));

                // 取出 a 和 b（半精度先转 float）
                const float a = to_f32(in[base + j]);
                const float b = to_f32(in[base + half + j]);

                // a' = a cos(phi) - b sin(phi)
                // b' = b cos(phi) + a sin(phi)
                const float ap = a * c - b * si;
                const float bp = b * c + a * si;

                // 写回 out（需要时cast回半精度）
                out[base + j] = from_f32<T>(ap);
                out[base + half + j] = from_f32<T>(bp);
            }
        }
    }
}

namespace llaisys::ops::cpu {

void rope(std::byte* out,
    const std::byte* in,
    const std::byte* pos_ids,
    llaisysDataType_t dtype,
    size_t seqlen,
    size_t nhead,
    size_t d,
    float theta) {
    // 使用 dtype 枚举做运行时分发：byte 指针强转成对应类型指针，再调用模板内核
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     seqlen, nhead, d, theta);
    default:
        // 未实现的 dtype（如 int8/int64 作为 in/out）直接报错
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}