#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// attn_val: [seqlen, nhead, dv]
// q:        [seqlen, nhead, d]
// k:        [total_len, nkvhead, d]
// v:        [total_len, nkvhead, dv]
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
                    float scale);
} // namespace llaisys::ops::cpu