#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <type_traits>

template <typename T>
static inline float to_f32(T x) {
    return llaisys::utils::cast<float>(x);
}

template <typename T>
static inline T from_f32(float x) {
    return llaisys::utils::cast<T>(x);
}

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            // half/bf16: promote to float for exp, then cast back
            float g = to_f32(gate[i]);
            float u = to_f32(up[i]);
            float s = 1.0f / (1.0f + std::exp(-g)); // sigmoid(g)
            float y = u * (g * s);                  // u * (g * sigmoid(g))
            out[i] = from_f32<T>(y);
        } else {
            // float path
            float g = gate[i];
            float u = up[i];
            float s = 1.0f / (1.0f + std::exp(-g));
            out[i] = static_cast<T>(u * (g * s));
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(gate),
                       reinterpret_cast<const float *>(up),
                       numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up),
                       numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up),
                       numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu