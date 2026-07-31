#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <type_traits>
#include <cstdint>

namespace {

// 将任意支持类型转成 float 参与累加：
// - 对 F16/BF16：先转 float 做乘加，最后再转回（更稳定、也符合 add 的实现思路）
// - 对 F32：直接 static_cast<float> 即可
template <typename T>
inline float to_f32(T v) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::cast<float>(v);
    } else {
        return static_cast<float>(v);
    }
}

// 将float结果转回目标类型T
template <typename T>
inline T from_f32(float v) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::cast<T>(v);
    } else {
        return static_cast<T>(v);
    }
}

// 朴素 Linear 核心：
// out[m, n] = sum_{k=0..K-1} in[m, k] * weight[n, k] + bias[n]
// 注意：weight 的形状是 [N, K]，即“每个输出通道一行”
template <typename T>
void linear_(T* out,
             const T* in,
             const T* weight,
             const T* bias,
             size_t M, size_t K, size_t N,
             bool has_bias) {
    // 假设所有张量都是 contiguous、row-major：
    // in:     [M, K]，第 m 行起点：in + m*K
    // weight: [N, K]，第 n 行起点：weight + n*K
    // out:    [M, N]，第 m 行起点：out + m*N
    for (size_t m = 0; m < M; ++m) {
        const T *in_row = in + m * K; // in[m, 0] 的地址
        T *out_row = out + m * N; // out[m, 0] 的地址

        for (size_t n = 0; n < N; ++n) {
            const T *w_row = weight + n * K; // weight[n, 0]的地址

            // 累加器用float:半精度更准确；f32也没问题
            float acc = has_bias ? to_f32(bias[n]) : 0.0f; 

            // 点积：in_row与w_row的长度为K
            for (size_t k = 0; k < K; ++k) {
                acc += to_f32(in_row[k]) * to_f32(w_row[k]); 
            }

            // 写回out
            out_row[n] = from_f32<T>(acc);
        }
    }
}
} // namespace

namespace llaisys::ops::cpu {

    void linear(std::byte* out,
        const std::byte* in,
        const std::byte* weight,
        const std::byte* bias, // 可为空
        llaisysDataType_t dtype,
        size_t M, size_t K, size_t N,
        bool has_bias) {
        // 用dtype枚举做运行时分发
        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return linear_(reinterpret_cast<float *>(out),
                           reinterpret_cast<const float *>(in),
                           reinterpret_cast<const float *>(weight),
                           reinterpret_cast<const float *>(bias),
                           M, K, N, has_bias);
        case LLAISYS_DTYPE_BF16:
            return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                           reinterpret_cast<const llaisys::bf16_t *>(in),
                           reinterpret_cast<const llaisys::bf16_t *>(weight),
                           reinterpret_cast<const llaisys::bf16_t *>(bias),
                           M, K, N, has_bias);
        case LLAISYS_DTYPE_F16:
            return linear_(
                reinterpret_cast<llaisys::fp16_t *>(out),
                reinterpret_cast<const llaisys::fp16_t *>(in),
                reinterpret_cast<const llaisys::fp16_t *>(weight),
                reinterpret_cast<const llaisys::fp16_t *>(bias),
                M, K, N, has_bias);
        default:
            // 其他 dtype（如 int8/int64）未实现
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    }

} // namespace llaisys::ops::cpu