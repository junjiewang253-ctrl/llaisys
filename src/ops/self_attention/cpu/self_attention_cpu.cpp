#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

// 将任意支持的标量类型转成 float（用于在 CPU 上做更稳定的累加/exp）
template <typename T>
static inline float to_f32(T x) {
    return llaisys::utils::cast<float>(x);
}

// 将 float 转回目标类型（float/f16/bf16）
template <typename T>
static inline T from_f32(float x) {
    return llaisys::utils::cast<T>(x);
}

/*
q:   [seqlen, nhead, d]
k:   [total_len, nkvhead, d]
v:   [total_len, nkvhead, dv]
out: [seqlen, nhead, dv]

实现目标对齐 test/ops/self_attention.py 中 torch_self_attention 的行为：
1) causal mask 使用 tril(diagonal=S-L)，其中 L=seqlen, S=total_len
   -> 允许 s <= t + (S - L)
2) GQA/MQA：torch 里用 repeat_interleave 把 kv head 扩展到 q head
   -> kv head 映射为 kh = h / (nhead / nkvhead)，而不是 h % nkvhead
*/
template <typename T>
static void self_attention_(T *out,
                            const T *q,
                            const T *k,
                            const T *v,
                            size_t seqlen,
                            size_t nhead,
                            size_t nkvhead,
                            size_t d,
                            size_t dv,
                            size_t total_len,
                            float scale) {
    if (nkvhead == 0) {
        return;
    }
    if (nhead % nkvhead != 0) {
        // 上层 op.cpp 会 ASSERT；这里避免未定义行为
        return;
    }

    const size_t group = nhead / nkvhead;

    // 对齐 torch 的 causal mask：
    // L = seqlen(qlen), S = total_len(kvlen)
    // 允许: s <= t + (S - L)
    const ptrdiff_t shift = static_cast<ptrdiff_t>(total_len) - static_cast<ptrdiff_t>(seqlen);

    // 复用 acc buffer：避免 (tq,h) 级别频繁 new/delete
    std::vector<float> acc(dv);

    for (size_t tq = 0; tq < seqlen; ++tq) {
        ptrdiff_t max_k_signed = static_cast<ptrdiff_t>(tq) + shift;

        if (max_k_signed < 0) {
            // 看不到任何 key -> 输出置 0
            for (size_t h = 0; h < nhead; ++h) {
                const size_t out_off = (tq * nhead + h) * dv;
                for (size_t j = 0; j < dv; ++j) {
                    out[out_off + j] = from_f32<T>(0.f);
                }
            }
            continue;
        }

        size_t max_k = static_cast<size_t>(max_k_signed);
        if (max_k >= total_len) {
            max_k = total_len - 1;
        }

        for (size_t h = 0; h < nhead; ++h) {
            const size_t kh = h / group;

            // q[tq, h, :]
            const T *q_ptr = q + (tq * nhead + h) * d;

            // 第 1 遍：找 max_logit
            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t tk = 0; tk <= max_k; ++tk) {
                const T *k_ptr = k + (tk * nkvhead + kh) * d;

                float dot = 0.f;
                for (size_t i = 0; i < d; ++i) {
                    dot += to_f32(q_ptr[i]) * to_f32(k_ptr[i]);
                }
                const float logit = dot * scale;
                if (logit > max_logit) {
                    max_logit = logit;
                }
            }

            // 第 2 遍：softmax + 加权求和 v
            std::fill(acc.begin(), acc.end(), 0.f);
            float denom = 0.f;

            for (size_t tk = 0; tk <= max_k; ++tk) {
                const T *k_ptr = k + (tk * nkvhead + kh) * d;

                float dot = 0.f;
                for (size_t i = 0; i < d; ++i) {
                    dot += to_f32(q_ptr[i]) * to_f32(k_ptr[i]);
                }
                const float logit = dot * scale;

                const float w = std::exp(logit - max_logit);
                denom += w;

                const T *v_ptr = v + (tk * nkvhead + kh) * dv;
                for (size_t j = 0; j < dv; ++j) {
                    acc[j] += w * to_f32(v_ptr[j]);
                }
            }

            const size_t out_off = (tq * nhead + h) * dv;

            // 极端情况下 denom 可能为 0（数值下溢）；这里做保护
            if (denom <= 0.f || !std::isfinite(denom)) {
                for (size_t j = 0; j < dv; ++j) {
                    out[out_off + j] = from_f32<T>(0.f);
                }
                continue;
            }

            const float inv_denom = 1.f / denom;
            for (size_t j = 0; j < dv; ++j) {
                out[out_off + j] = from_f32<T>(acc[j] * inv_denom);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    llaisysDataType_t dtype,
                    size_t seqlen,
                    size_t nhead,
                    size_t nkvhead,
                    size_t d,
                    size_t dv,
                    size_t total_len,
                    float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               seqlen, nhead, nkvhead, d, dv, total_len, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               seqlen, nhead, nkvhead, d, dv, total_len, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               seqlen, nhead, nkvhead, d, dv, total_len, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu