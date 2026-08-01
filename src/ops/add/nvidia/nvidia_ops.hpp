#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void add(std::byte *out, const std::byte *a, const std::byte *b,
         llaisysDataType_t dtype, size_t numel, llaisysStream_t stream);
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t dtype,
               const size_t *shape, const ptrdiff_t *out_strides,
               const ptrdiff_t *in_strides, size_t ndim, llaisysStream_t stream);
void argmax(std::byte *index, std::byte *value, const std::byte *input,
            llaisysDataType_t dtype, size_t numel, llaisysStream_t stream);
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t n, size_t d, size_t vocab,
               llaisysStream_t stream);
void linear(std::byte *out, const std::byte *input, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype,
            size_t m, size_t k, size_t n, bool has_bias,
            llaisysStream_t stream);
void rmsNorm(std::byte *out, const std::byte *input, const std::byte *weight,
             llaisysDataType_t dtype, size_t m, size_t d, float eps,
             llaisysStream_t stream);
void rope(std::byte *out, const std::byte *input, const std::byte *positions,
          llaisysDataType_t dtype, size_t seqlen, size_t nhead, size_t d,
          float theta, llaisysStream_t stream);
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel, llaisysStream_t stream);
void selfAttention(std::byte *out, const std::byte *q, const std::byte *k,
                   const std::byte *v, llaisysDataType_t dtype,
                   size_t seqlen, size_t nhead, size_t nkvhead,
                   size_t d, size_t dv, size_t total_len, float scale,
                   llaisysStream_t stream);
} // namespace llaisys::ops::nvidia
