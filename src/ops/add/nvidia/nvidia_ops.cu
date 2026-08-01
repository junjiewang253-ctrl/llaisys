#include "nvidia_ops.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
namespace {
constexpr unsigned int BLOCK = 256;
constexpr size_t MAX_NDIM = 8;

void checkCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

cudaStream_t cudaStream(llaisysStream_t stream) {
    if (stream == nullptr) {
        throw std::invalid_argument("null CUDA operation stream");
    }
    return reinterpret_cast<cudaStream_t>(stream);
}

unsigned int blocks(size_t count) {
    return static_cast<unsigned int>((count + BLOCK - 1) / BLOCK);
}

template <typename T> __device__ float toFloat(T value);
template <> __device__ float toFloat<float>(float value) { return value; }
template <> __device__ float toFloat<__half>(__half value) {
    return __half2float(value);
}
template <> __device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T> __device__ T fromFloat(float value);
template <> __device__ float fromFloat<float>(float value) { return value; }
template <> __device__ __half fromFloat<__half>(float value) {
    return __float2half_rn(value);
}
template <> __device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T> __device__ float dtypeRound(float value) {
    return toFloat(fromFloat<T>(value));
}
template <> __device__ float dtypeRound<float>(float value) { return value; }

template <typename T>
__global__ void llaisys_cuda_add_kernel(
    T *out, const T *a, const T *b, size_t count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = fromFloat<T>(toFloat(a[index]) + toFloat(b[index]));
    }
}

struct RearrangeMeta {
    size_t shape[MAX_NDIM];
    ptrdiff_t out_strides[MAX_NDIM];
    ptrdiff_t in_strides[MAX_NDIM];
    size_t ndim;
};

template <typename T>
__global__ void llaisys_cuda_rearrange_kernel(
    T *out, const T *in, RearrangeMeta meta, size_t count) {
    size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= count) return;
    size_t remaining = linear;
    ptrdiff_t out_offset = 0;
    ptrdiff_t in_offset = 0;
    for (size_t axis = meta.ndim; axis != 0; --axis) {
        const size_t dim = axis - 1;
        const size_t coordinate = remaining % meta.shape[dim];
        remaining /= meta.shape[dim];
        out_offset += static_cast<ptrdiff_t>(coordinate) * meta.out_strides[dim];
        in_offset += static_cast<ptrdiff_t>(coordinate) * meta.in_strides[dim];
    }
    out[out_offset] = in[in_offset];
}

template <typename T>
__global__ void llaisys_cuda_argmax_kernel(
    int64_t *out_index, T *out_value, const T *input, size_t count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    size_t best_index = 0;
    float best = toFloat(input[0]);
    for (size_t index = 1; index < count; ++index) {
        const float value = toFloat(input[index]);
        if (value > best) {
            best = value;
            best_index = index;
        }
    }
    *out_index = static_cast<int64_t>(best_index);
    *out_value = input[best_index];
}

template <typename T>
__global__ void llaisys_cuda_embedding_kernel(
    T *out, const int64_t *index, const T *weight,
    size_t n, size_t d, size_t vocab) {
    const size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= n * d) return;
    const size_t row = linear / d;
    const size_t column = linear % d;
    const int64_t source = index[row];
    if (source >= 0 && static_cast<size_t>(source) < vocab) {
        out[linear] = weight[static_cast<size_t>(source) * d + column];
    }
}

template <typename T>
__global__ void llaisys_cuda_linear_kernel(
    T *out, const T *input, const T *weight, const T *bias,
    size_t m, size_t k, size_t n, bool has_bias) {
    const size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= m * n) return;
    const size_t row = linear / n;
    const size_t column = linear % n;
    float partial[8] = {};
    for (size_t inner = 0; inner < k; ++inner) {
        partial[inner % 8] += toFloat(input[row * k + inner])
                            * toFloat(weight[column * k + inner]);
    }
    float result = has_bias ? toFloat(bias[column]) : 0.0f;
    for (float value : partial) result += value;
    out[linear] = fromFloat<T>(result);
}

template <typename T>
__global__ void llaisys_cuda_rms_norm_kernel(
    T *out, const T *input, const T *weight, size_t m, size_t d, float eps) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    float sum = 0.0f;
    for (size_t column = 0; column < d; ++column) {
        const float value = toFloat(input[row * d + column]);
        sum += value * value;
    }
    const float inverse = rsqrtf(sum / static_cast<float>(d) + eps);
    for (size_t column = 0; column < d; ++column) {
        float normalized = toFloat(input[row * d + column]) * inverse;
        normalized = dtypeRound<T>(normalized);
        out[row * d + column] = fromFloat<T>(
            normalized * toFloat(weight[column]));
    }
}

template <typename T>
__global__ void llaisys_cuda_rope_kernel(
    T *out, const T *input, const int64_t *positions,
    size_t seqlen, size_t nhead, size_t d, float theta) {
    const size_t half = d / 2;
    const size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= seqlen * nhead * half) return;
    const size_t pair = linear % half;
    const size_t head_token = linear / half;
    const size_t token = head_token / nhead;
    const size_t base = head_token * d;
    const float exponent = 2.0f * static_cast<float>(pair) / static_cast<float>(d);
    const float angle = static_cast<float>(positions[token]) / powf(theta, exponent);
    const float sine = sinf(angle);
    const float cosine = cosf(angle);
    const float a = toFloat(input[base + pair]);
    const float b = toFloat(input[base + half + pair]);
    out[base + pair] = fromFloat<T>(a * cosine - b * sine);
    out[base + half + pair] = fromFloat<T>(b * cosine + a * sine);
}

template <typename T>
__global__ void llaisys_cuda_swiglu_kernel(
    T *out, const T *gate, const T *up, size_t count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float g = toFloat(gate[index]);
    const float u = toFloat(up[index]);
    out[index] = fromFloat<T>(u * g / (1.0f + expf(-g)));
}

template <typename T>
__global__ void llaisys_cuda_self_attention_kernel(
    T *out, const T *q, const T *k, const T *v,
    size_t seqlen, size_t nhead, size_t nkvhead,
    size_t d, size_t dv, size_t total_len, float scale) {
    const size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= seqlen * nhead * dv) return;
    const size_t value_dim = linear % dv;
    const size_t head_token = linear / dv;
    const size_t head = head_token % nhead;
    const size_t token = head_token / nhead;
    const size_t group = nhead / nkvhead;
    const size_t kv_head = head / group;
    const size_t max_key = token + total_len - seqlen;
    const T *query = q + (token * nhead + head) * d;
    float max_logit = -INFINITY;
    for (size_t key_index = 0; key_index <= max_key; ++key_index) {
        const T *key = k + (key_index * nkvhead + kv_head) * d;
        float dot = 0.0f;
        for (size_t inner = 0; inner < d; ++inner) {
            dot += toFloat(query[inner]) * toFloat(key[inner]);
        }
        const float logit = dtypeRound<T>(dtypeRound<T>(dot) * scale);
        max_logit = fmaxf(max_logit, logit);
    }
    float denominator = 0.0f;
    for (size_t key_index = 0; key_index <= max_key; ++key_index) {
        const T *key = k + (key_index * nkvhead + kv_head) * d;
        float dot = 0.0f;
        for (size_t inner = 0; inner < d; ++inner) {
            dot += toFloat(query[inner]) * toFloat(key[inner]);
        }
        const float logit = dtypeRound<T>(dtypeRound<T>(dot) * scale);
        denominator += expf(logit - max_logit);
    }
    float result = 0.0f;
    for (size_t key_index = 0; key_index <= max_key; ++key_index) {
        const T *key = k + (key_index * nkvhead + kv_head) * d;
        float dot = 0.0f;
        for (size_t inner = 0; inner < d; ++inner) {
            dot += toFloat(query[inner]) * toFloat(key[inner]);
        }
        const float logit = dtypeRound<T>(dtypeRound<T>(dot) * scale);
        const float probability = dtypeRound<T>(
            expf(logit - max_logit) / denominator);
        result += probability * toFloat(
            v[(key_index * nkvhead + kv_head) * dv + value_dim]);
    }
    out[linear] = fromFloat<T>(result);
}

template <typename T>
void launchAdd(std::byte *out, const std::byte *a, const std::byte *b,
               size_t count, cudaStream_t stream) {
    if (count == 0) return;
    llaisys_cuda_add_kernel<<<blocks(count), BLOCK, 0, stream>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(a),
        reinterpret_cast<const T *>(b), count);
}

template <typename T>
void launchRearrange(std::byte *out, const std::byte *in,
                     const RearrangeMeta &meta, size_t count,
                     cudaStream_t stream) {
    if (count == 0) return;
    llaisys_cuda_rearrange_kernel<<<blocks(count), BLOCK, 0, stream>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), meta, count);
}

template <typename T>
void launchArgmax(std::byte *index, std::byte *value, const std::byte *input,
                  size_t count, cudaStream_t stream) {
    llaisys_cuda_argmax_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<int64_t *>(index), reinterpret_cast<T *>(value),
        reinterpret_cast<const T *>(input), count);
}

template <typename T>
void launchEmbedding(std::byte *out, const std::byte *index,
                     const std::byte *weight, size_t n, size_t d, size_t vocab,
                     cudaStream_t stream) {
    const size_t count = n * d;
    if (count == 0) return;
    llaisys_cuda_embedding_kernel<<<blocks(count), BLOCK, 0, stream>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const int64_t *>(index),
        reinterpret_cast<const T *>(weight), n, d, vocab);
}

template <typename T>
void launchLinear(std::byte *out, const std::byte *input,
                  const std::byte *weight, const std::byte *bias,
                  size_t m, size_t k, size_t n, bool has_bias,
                  cudaStream_t stream) {
    const size_t count = m * n;
    if (count == 0) return;
    llaisys_cuda_linear_kernel<<<blocks(count), BLOCK, 0, stream>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(input),
        reinterpret_cast<const T *>(weight), reinterpret_cast<const T *>(bias),
        m, k, n, has_bias);
}

template <typename T>
void launchRmsNorm(std::byte *out, const std::byte *input,
                   const std::byte *weight, size_t m, size_t d, float eps,
                   cudaStream_t stream) {
    if (m == 0) return;
    llaisys_cuda_rms_norm_kernel<<<blocks(m), BLOCK, 0, stream>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(input),
        reinterpret_cast<const T *>(weight), m, d, eps);
}

template <typename T>
void launchRope(std::byte *out, const std::byte *input,
                const std::byte *positions, size_t seqlen, size_t nhead,
                size_t d, float theta, cudaStream_t stream) {
    const size_t count = seqlen * nhead * (d / 2);
    if (count == 0) return;
    llaisys_cuda_rope_kernel<<<blocks(count), BLOCK, 0, stream>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(input),
        reinterpret_cast<const int64_t *>(positions), seqlen, nhead, d, theta);
}

template <typename T>
void launchSwiglu(std::byte *out, const std::byte *gate, const std::byte *up,
                  size_t count, cudaStream_t stream) {
    if (count == 0) return;
    llaisys_cuda_swiglu_kernel<<<blocks(count), BLOCK, 0, stream>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up), count);
}

template <typename T>
void launchSelfAttention(std::byte *out, const std::byte *q,
                         const std::byte *k, const std::byte *v,
                         size_t seqlen, size_t nhead, size_t nkvhead,
                         size_t d, size_t dv, size_t total_len, float scale,
                         cudaStream_t stream) {
    const size_t count = seqlen * nhead * dv;
    if (count == 0) return;
    llaisys_cuda_self_attention_kernel<<<blocks(count), BLOCK, 0, stream>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k), reinterpret_cast<const T *>(v),
        seqlen, nhead, nkvhead, d, dv, total_len, scale);
}

#define DISPATCH_FLOAT_TYPES(call) \
    switch (dtype) { \
    case LLAISYS_DTYPE_F32: call(float); break; \
    case LLAISYS_DTYPE_F16: call(__half); break; \
    case LLAISYS_DTYPE_BF16: call(__nv_bfloat16); break; \
    default: throw std::invalid_argument("unsupported CUDA operation dtype"); \
    }
} // namespace

void add(std::byte *out, const std::byte *a, const std::byte *b,
         llaisysDataType_t dtype, size_t count, llaisysStream_t stream) {
#define CALL(T) launchAdd<T>(out, a, b, count, cudaStream(stream))
    DISPATCH_FLOAT_TYPES(CALL);
#undef CALL
    checkCuda(cudaPeekAtLastError(), "CUDA add launch");
}

void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t dtype,
               const size_t *shape, const ptrdiff_t *out_strides,
               const ptrdiff_t *in_strides, size_t ndim, llaisysStream_t stream) {
    if (ndim > MAX_NDIM) throw std::invalid_argument("CUDA rearrange rank exceeds 8");
    RearrangeMeta meta{};
    meta.ndim = ndim;
    size_t count = 1;
    for (size_t axis = 0; axis < ndim; ++axis) {
        meta.shape[axis] = shape[axis];
        meta.out_strides[axis] = out_strides[axis];
        meta.in_strides[axis] = in_strides[axis];
        count *= shape[axis];
    }
#define CALL(T) launchRearrange<T>(out, in, meta, count, cudaStream(stream))
    if (dtype == LLAISYS_DTYPE_I64) {
        launchRearrange<int64_t>(out, in, meta, count, cudaStream(stream));
    } else {
        DISPATCH_FLOAT_TYPES(CALL);
    }
#undef CALL
    checkCuda(cudaPeekAtLastError(), "CUDA rearrange launch");
}

void argmax(std::byte *index, std::byte *value, const std::byte *input,
            llaisysDataType_t dtype, size_t count, llaisysStream_t stream) {
#define CALL(T) launchArgmax<T>(index, value, input, count, cudaStream(stream))
    DISPATCH_FLOAT_TYPES(CALL);
#undef CALL
    checkCuda(cudaPeekAtLastError(), "CUDA argmax launch");
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t n, size_t d, size_t vocab,
               llaisysStream_t stream) {
#define CALL(T) launchEmbedding<T>(out, index, weight, n, d, vocab, cudaStream(stream))
    DISPATCH_FLOAT_TYPES(CALL);
#undef CALL
    checkCuda(cudaPeekAtLastError(), "CUDA embedding launch");
}

void linear(std::byte *out, const std::byte *input, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype,
            size_t m, size_t k, size_t n, bool has_bias,
            llaisysStream_t stream) {
#define CALL(T) launchLinear<T>(out, input, weight, bias, m, k, n, has_bias, cudaStream(stream))
    DISPATCH_FLOAT_TYPES(CALL);
#undef CALL
    checkCuda(cudaPeekAtLastError(), "CUDA linear launch");
}

void rmsNorm(std::byte *out, const std::byte *input, const std::byte *weight,
             llaisysDataType_t dtype, size_t m, size_t d, float eps,
             llaisysStream_t stream) {
#define CALL(T) launchRmsNorm<T>(out, input, weight, m, d, eps, cudaStream(stream))
    DISPATCH_FLOAT_TYPES(CALL);
#undef CALL
    checkCuda(cudaPeekAtLastError(), "CUDA RMSNorm launch");
}

void rope(std::byte *out, const std::byte *input, const std::byte *positions,
          llaisysDataType_t dtype, size_t seqlen, size_t nhead, size_t d,
          float theta, llaisysStream_t stream) {
#define CALL(T) launchRope<T>(out, input, positions, seqlen, nhead, d, theta, cudaStream(stream))
    DISPATCH_FLOAT_TYPES(CALL);
#undef CALL
    checkCuda(cudaPeekAtLastError(), "CUDA RoPE launch");
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t count, llaisysStream_t stream) {
#define CALL(T) launchSwiglu<T>(out, gate, up, count, cudaStream(stream))
    DISPATCH_FLOAT_TYPES(CALL);
#undef CALL
    checkCuda(cudaPeekAtLastError(), "CUDA SwiGLU launch");
}

void selfAttention(std::byte *out, const std::byte *q, const std::byte *k,
                   const std::byte *v, llaisysDataType_t dtype,
                   size_t seqlen, size_t nhead, size_t nkvhead,
                   size_t d, size_t dv, size_t total_len, float scale,
                   llaisysStream_t stream) {
#define CALL(T) launchSelfAttention<T>(out, q, k, v, seqlen, nhead, nkvhead, d, dv, total_len, scale, cudaStream(stream))
    DISPATCH_FLOAT_TYPES(CALL);
#undef CALL
    checkCuda(cudaPeekAtLastError(), "CUDA self-attention launch");
}
} // namespace llaisys::ops::nvidia
