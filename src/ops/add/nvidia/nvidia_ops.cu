#include "nvidia_ops.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

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

void checkCublas(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + ": cuBLAS status " +
            std::to_string(static_cast<int>(status)));
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

size_t checkedMultiply(size_t left, size_t right, const char *operation) {
    if (left != 0 && right > SIZE_MAX / left) {
        throw std::invalid_argument(
            std::string(operation) + " size overflow");
    }
    return left * right;
}

size_t checkedAdd(size_t left, size_t right, const char *operation) {
    if (right > SIZE_MAX - left) {
        throw std::invalid_argument(
            std::string(operation) + " size overflow");
    }
    return left + right;
}

uint32_t pointerAlignment(const void *pointer) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    uint32_t alignment = 256;
    while (alignment > 1 && address % alignment != 0) alignment /= 2;
    return alignment;
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
__global__ void llaisys_cuda_linear_bias_kernel(
    T *out, const T *bias, size_t count, size_t columns) {
    const size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear < count) {
        out[linear] = bias[linear % columns];
    }
}

template <typename T>
__global__ void llaisys_cuda_rms_norm_kernel(
    T *out, const T *input, const T *weight, size_t m, size_t d,
    float mean_factor, float eps) {
    const size_t row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned int partial_index =
        threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int partial_base = threadIdx.y * blockDim.x;
    __shared__ float partial[512];
    float accumulators[4] = {};
    if (row < m) {
        const size_t vector_count = d / 4;
        for (size_t vector = threadIdx.x; vector < vector_count;
             vector += blockDim.x) {
#pragma unroll
            for (size_t component = 0; component < 4; ++component) {
                const float value = toFloat(
                    input[row * d + vector * 4 + component]);
                const float squared = __fmul_rn(value, value);
                accumulators[component] =
                    __fadd_rn(accumulators[component], squared);
            }
        }
        const size_t tail = vector_count * 4 + threadIdx.x;
        if (tail < d) {
            const float value = toFloat(input[row * d + tail]);
            accumulators[0] = __fadd_rn(
                accumulators[0], __fmul_rn(value, value));
        }
    }
    float sum = accumulators[0] + accumulators[1];
    sum += accumulators[2];
    sum += accumulators[3];
    partial[partial_index] = sum;
    for (unsigned int offset = blockDim.x / 2; offset >= 32; offset /= 2) {
        __syncthreads();
        if (threadIdx.x < offset) {
            sum += partial[partial_base + threadIdx.x + offset];
            partial[partial_index] = sum;
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
#pragma unroll
        for (unsigned int offset = 1; offset < 32; offset *= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) partial[partial_base] = sum;
    }
    __syncthreads();
    if (row >= m) return;
    const float mean = __fmul_rn(partial[partial_base], mean_factor);
    const float variance = __fadd_rn(mean, eps);
    const float inverse = rsqrtf(variance);
    for (size_t column = threadIdx.x; column < d; column += blockDim.x) {
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
    const float sine = dtypeRound<T>(sinf(angle));
    const float cosine = dtypeRound<T>(cosf(angle));
    const float a = toFloat(input[base + pair]);
    const float b = toFloat(input[base + half + pair]);
    const float a_cosine = dtypeRound<T>(a * cosine);
    const float b_cosine = dtypeRound<T>(b * cosine);
    const float negative_b_sine = dtypeRound<T>((-b) * sine);
    const float a_sine = dtypeRound<T>(a * sine);
    out[base + pair] = fromFloat<T>(a_cosine + negative_b_sine);
    out[base + half + pair] = fromFloat<T>(b_cosine + a_sine);
}

template <typename T>
__global__ void llaisys_cuda_swiglu_kernel(
    T *out, const T *gate, const T *up, size_t count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float g = toFloat(gate[index]);
    const float u = toFloat(up[index]);
    const float silu = dtypeRound<T>(g / (1.0f + expf(-g)));
    out[index] = fromFloat<T>(u * silu);
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
__global__ void llaisys_cuda_attention_softmax_kernel(
    T *probabilities, size_t seqlen, size_t total_len, float scale) {
    const size_t token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= seqlen) return;
    const size_t max_key = token + total_len - seqlen;
    T *row = probabilities + token * total_len;
    float maximum = -INFINITY;
    for (size_t key = 0; key <= max_key; ++key) {
        const float logit = dtypeRound<T>(toFloat(row[key]) * scale);
        row[key] = fromFloat<T>(logit);
        maximum = fmaxf(maximum, logit);
    }
    float denominator = 0.0f;
    for (size_t key = 0; key <= max_key; ++key) {
        denominator += expf(toFloat(row[key]) - maximum);
    }
    for (size_t key = 0; key <= max_key; ++key) {
        row[key] = fromFloat<T>(
            expf(toFloat(row[key]) - maximum) / denominator);
    }
    for (size_t key = max_key + 1; key < total_len; ++key) {
        row[key] = fromFloat<T>(0.0f);
    }
}

template <typename T, int LOG2_ELEMENTS>
__global__ void llaisys_cuda_attention_softmax_warp_kernel(
    T *probabilities, size_t seqlen, size_t total_len, float scale) {
    constexpr int next_power_of_two = 1 << LOG2_ELEMENTS;
    constexpr int warp_size = next_power_of_two < 32 ? next_power_of_two : 32;
    constexpr int warp_iterations = next_power_of_two / warp_size;
    constexpr int warp_batch = next_power_of_two <= 128 ? 2 : 1;

    const size_t first_token =
        (blockDim.y * blockIdx.x + threadIdx.y) * warp_batch;
    const size_t local_index = threadIdx.x;
    const size_t remaining_tokens =
        first_token < seqlen ? seqlen - first_token : 0;
    const size_t local_tokens =
        remaining_tokens < static_cast<size_t>(warp_batch)
            ? remaining_tokens
            : static_cast<size_t>(warp_batch);
    float elements[warp_batch][warp_iterations];

#pragma unroll
    for (int batch = 0; batch < warp_batch; ++batch) {
        const size_t token = first_token + batch;
        const size_t max_key = token + total_len - seqlen;
#pragma unroll
        for (int iteration = 0; iteration < warp_iterations; ++iteration) {
            const size_t key = local_index + iteration * warp_size;
            if (static_cast<size_t>(batch) < local_tokens &&
                key < total_len && key <= max_key) {
                T *row = probabilities + token * total_len;
                elements[batch][iteration] =
                    dtypeRound<T>(toFloat(row[key]) * scale);
            } else {
                elements[batch][iteration] = -INFINITY;
            }
        }
    }

    float maximum[warp_batch];
#pragma unroll
    for (int batch = 0; batch < warp_batch; ++batch) {
        maximum[batch] = elements[batch][0];
#pragma unroll
        for (int iteration = 0; iteration < warp_iterations; ++iteration) {
            maximum[batch] = maximum[batch] > elements[batch][iteration]
                                 ? maximum[batch]
                                 : elements[batch][iteration];
        }
    }
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int batch = 0; batch < warp_batch; ++batch) {
            const float other = __shfl_xor_sync(
                0xffffffff, maximum[batch], offset, warp_size);
            maximum[batch] =
                maximum[batch] < other ? other : maximum[batch];
        }
    }

    float denominator[warp_batch] = {};
#pragma unroll
    for (int batch = 0; batch < warp_batch; ++batch) {
#pragma unroll
        for (int iteration = 0; iteration < warp_iterations; ++iteration) {
            elements[batch][iteration] =
                expf(elements[batch][iteration] - maximum[batch]);
            denominator[batch] += elements[batch][iteration];
        }
    }
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int batch = 0; batch < warp_batch; ++batch) {
            denominator[batch] += __shfl_xor_sync(
                0xffffffff, denominator[batch], offset, warp_size);
        }
    }

#pragma unroll
    for (int batch = 0; batch < warp_batch; ++batch) {
        if (static_cast<size_t>(batch) >= local_tokens) break;
        T *row = probabilities + (first_token + batch) * total_len;
#pragma unroll
        for (int iteration = 0; iteration < warp_iterations; ++iteration) {
            const size_t key = local_index + iteration * warp_size;
            if (key < total_len) {
                row[key] = fromFloat<T>(
                    elements[batch][iteration] / denominator[batch]);
            }
        }
    }
}

template <typename T, int LOG2_ELEMENTS>
void launchAttentionSoftmaxWarp(
    T *probabilities, size_t seqlen, size_t total_len, float scale,
    cudaStream_t stream) {
    constexpr int next_power_of_two = 1 << LOG2_ELEMENTS;
    constexpr int warp_size = next_power_of_two < 32 ? next_power_of_two : 32;
    constexpr int warp_batch = next_power_of_two <= 128 ? 2 : 1;
    constexpr int threads_per_block = 128;
    constexpr int warps_per_block = threads_per_block / warp_size;
    constexpr int tokens_per_block = warps_per_block * warp_batch;
    const unsigned int block_count = static_cast<unsigned int>(
        (seqlen + tokens_per_block - 1) / tokens_per_block);
    const dim3 threads(warp_size, warps_per_block, 1);
    llaisys_cuda_attention_softmax_warp_kernel<T, LOG2_ELEMENTS>
        <<<block_count, threads, 0, stream>>>(
            probabilities, seqlen, total_len, scale);
}

template <typename T>
void launchAttentionSoftmax(
    T *probabilities, size_t seqlen, size_t total_len, float scale,
    cudaStream_t stream) {
#define SOFTMAX_CASE(LOG2) \
    case (static_cast<size_t>(1) << (LOG2)): \
        launchAttentionSoftmaxWarp<T, LOG2>( \
            probabilities, seqlen, total_len, scale, stream); \
        return
    size_t next_power_of_two = 1;
    while (next_power_of_two < total_len && next_power_of_two < 1024) {
        next_power_of_two *= 2;
    }
    switch (next_power_of_two) {
        SOFTMAX_CASE(0);
        SOFTMAX_CASE(1);
        SOFTMAX_CASE(2);
        SOFTMAX_CASE(3);
        SOFTMAX_CASE(4);
        SOFTMAX_CASE(5);
        SOFTMAX_CASE(6);
        SOFTMAX_CASE(7);
        SOFTMAX_CASE(8);
        SOFTMAX_CASE(9);
        SOFTMAX_CASE(10);
        default: break;
    }
#undef SOFTMAX_CASE
    llaisys_cuda_attention_softmax_kernel<<<
        blocks(seqlen), BLOCK, 0, stream>>>(
            probabilities, seqlen, total_len, scale);
}

template <typename T>
__global__ void llaisys_cuda_repeat_kv_heads_kernel(
    T *out, const T *input, size_t total_len, size_t nhead,
    size_t nkvhead, size_t width) {
    const size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = nhead * total_len * width;
    if (linear >= count) return;
    const size_t column = linear % width;
    const size_t head_token = linear / width;
    const size_t token = head_token % total_len;
    const size_t head = head_token / total_len;
    const size_t kv_head = head / (nhead / nkvhead);
    out[linear] = input[(token * nkvhead + kv_head) * width + column];
}

template <typename T>
__global__ void llaisys_cuda_attention_head_to_token_kernel(
    T *out, const T *input, size_t seqlen, size_t nhead, size_t width) {
    const size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = seqlen * nhead * width;
    if (linear >= count) return;
    const size_t column = linear % width;
    const size_t head_token = linear / width;
    const size_t head = head_token % nhead;
    const size_t token = head_token / nhead;
    out[linear] = input[(head * seqlen + token) * width + column];
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
cudaDataType_t cudaDataType();
template <> cudaDataType_t cudaDataType<float>() { return CUDA_R_32F; }
template <> cudaDataType_t cudaDataType<__half>() { return CUDA_R_16F; }
template <> cudaDataType_t cudaDataType<__nv_bfloat16>() { return CUDA_R_16BF; }

void launchLinearBiasBf16Lt(
    __nv_bfloat16 *out, const __nv_bfloat16 *input,
    const __nv_bfloat16 *weight, const __nv_bfloat16 *bias,
    size_t m, size_t k, size_t n, cudaStream_t stream) {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t weight_layout = nullptr;
    cublasLtMatrixLayout_t input_layout = nullptr;
    cublasLtMatrixLayout_t output_layout = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    void *workspace = nullptr;
    constexpr size_t workspace_bytes = 1024 * 1024;
    auto cleanup = [&]() {
        if (workspace != nullptr) cudaFreeAsync(workspace, stream);
        if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
        if (output_layout != nullptr) cublasLtMatrixLayoutDestroy(output_layout);
        if (input_layout != nullptr) cublasLtMatrixLayoutDestroy(input_layout);
        if (weight_layout != nullptr) cublasLtMatrixLayoutDestroy(weight_layout);
        if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
        if (handle != nullptr) cublasLtDestroy(handle);
    };
    try {
        checkCublas(cublasLtCreate(&handle), "cublasLtCreate");
        checkCublas(
            cublasLtMatmulDescCreate(
                &operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "cublasLtMatmulDescCreate");
        const cublasOperation_t transpose = CUBLAS_OP_T;
        const cublasOperation_t identity = CUBLAS_OP_N;
        checkCublas(
            cublasLtMatmulDescSetAttribute(
                operation, CUBLASLT_MATMUL_DESC_TRANSA,
                &transpose, sizeof(transpose)),
            "cublasLtMatmulDescSetAttribute TRANSA");
        checkCublas(
            cublasLtMatmulDescSetAttribute(
                operation, CUBLASLT_MATMUL_DESC_TRANSB,
                &identity, sizeof(identity)),
            "cublasLtMatmulDescSetAttribute TRANSB");
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        checkCublas(
            cublasLtMatmulDescSetAttribute(
                operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epilogue, sizeof(epilogue)),
            "cublasLtMatmulDescSetAttribute EPILOGUE");
        const void *bias_pointer = bias;
        checkCublas(
            cublasLtMatmulDescSetAttribute(
                operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias_pointer, sizeof(bias_pointer)),
            "cublasLtMatmulDescSetAttribute BIAS_POINTER");
        checkCublas(
            cublasLtMatrixLayoutCreate(
                &weight_layout, CUDA_R_16BF, k, n, k),
            "cublasLtMatrixLayoutCreate weight");
        checkCublas(
            cublasLtMatrixLayoutCreate(
                &input_layout, CUDA_R_16BF, k, m, k),
            "cublasLtMatrixLayoutCreate input");
        checkCublas(
            cublasLtMatrixLayoutCreate(
                &output_layout, CUDA_R_16BF, n, m, n),
            "cublasLtMatrixLayoutCreate output");
        checkCublas(
            cublasLtMatmulPreferenceCreate(&preference),
            "cublasLtMatmulPreferenceCreate");
        checkCublas(
            cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_bytes, sizeof(workspace_bytes)),
            "cublasLtMatmulPreferenceSetAttribute workspace");
        const struct {
            cublasLtMatmulPreferenceAttributes_t attribute;
            const void *pointer;
        } alignments[] = {
            {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, weight},
            {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, input},
            {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, out},
            {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, out},
        };
        for (const auto &entry : alignments) {
            const uint32_t alignment = pointerAlignment(entry.pointer);
            checkCublas(
                cublasLtMatmulPreferenceSetAttribute(
                    preference, entry.attribute,
                    &alignment, sizeof(alignment)),
                "cublasLtMatmulPreferenceSetAttribute alignment");
        }
        cublasLtMatmulHeuristicResult_t heuristic{};
        int returned = 0;
        checkCublas(
            cublasLtMatmulAlgoGetHeuristic(
                handle, operation, weight_layout, input_layout,
                output_layout, output_layout, preference, 1,
                &heuristic, &returned),
            "cublasLtMatmulAlgoGetHeuristic");
        if (returned != 1) {
            throw std::runtime_error("no cuBLASLt BF16 biased linear algorithm");
        }
        checkCuda(
            cudaMallocAsync(&workspace, workspace_bytes, stream),
            "cudaMallocAsync cuBLASLt linear workspace");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        checkCublas(
            cublasLtMatmul(
                handle, operation, &alpha,
                weight, weight_layout, input, input_layout,
                &beta, out, output_layout, out, output_layout,
                &heuristic.algo, workspace, workspace_bytes, stream),
            "cublasLtMatmul BF16 biased linear");
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}

template <typename T>
void launchLinear(std::byte *out, const std::byte *input,
                  const std::byte *weight, const std::byte *bias,
                  size_t m, size_t k, size_t n, bool has_bias,
                  cudaStream_t stream) {
    const size_t count = m * n;
    if (count == 0) return;
    if (m > static_cast<size_t>(INT_MAX) ||
        k > static_cast<size_t>(INT_MAX) ||
        n > static_cast<size_t>(INT_MAX)) {
        throw std::invalid_argument("CUDA linear dimension exceeds cuBLAS int range");
    }
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        if (has_bias) {
            launchLinearBiasBf16Lt(
                reinterpret_cast<T *>(out),
                reinterpret_cast<const T *>(input),
                reinterpret_cast<const T *>(weight),
                reinterpret_cast<const T *>(bias), m, k, n, stream);
            return;
        }
    }
    if (has_bias) {
        llaisys_cuda_linear_bias_kernel<<<blocks(count), BLOCK, 0, stream>>>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(bias),
            count, n);
        checkCuda(cudaPeekAtLastError(), "CUDA linear bias launch");
    }
    cublasHandle_t handle = nullptr;
    checkCublas(cublasCreate(&handle), "cublasCreate");
    try {
        checkCublas(cublasSetStream(handle, stream), "cublasSetStream");
        const float alpha = 1.0f;
        const float beta = has_bias ? 1.0f : 0.0f;
        const auto data_type = cudaDataType<T>();
        checkCublas(
            cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                &alpha, weight, data_type, static_cast<int>(k),
                input, data_type, static_cast<int>(k),
                &beta, out, data_type, static_cast<int>(n),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmEx");
    } catch (...) {
        cublasDestroy(handle);
        throw;
    }
    checkCublas(cublasDestroy(handle), "cublasDestroy");
}

template <typename T>
void launchRmsNorm(std::byte *out, const std::byte *input,
                   const std::byte *weight, size_t m, size_t d, float eps,
                   cudaStream_t stream) {
    if (m == 0) return;
    unsigned int width = m >= 16 ? 32 : (m >= 8 ? 64 : 128);
    unsigned int height = 1;
    if (d > 128) {
        unsigned int output_power = 1;
        while (output_power < 16 && output_power * 2 <= m) {
            output_power *= 2;
        }
        height = output_power;
        width = 512 / height;
        if (width > 128) width = 128;
        if (width > 32 && height >= 16) width = 32;
    }
    const unsigned int grid =
        static_cast<unsigned int>((m + height - 1) / height);
    const float mean_factor = 1.0f / static_cast<float>(d);
    llaisys_cuda_rms_norm_kernel<<<grid, dim3(width, height), 0, stream>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(input),
        reinterpret_cast<const T *>(weight), m, d, mean_factor, eps);
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
    if (seqlen > static_cast<size_t>(INT_MAX) ||
        total_len > static_cast<size_t>(INT_MAX) ||
        nhead > static_cast<size_t>(INT_MAX) ||
        nkvhead > static_cast<size_t>(INT_MAX) ||
        d > static_cast<size_t>(INT_MAX) ||
        dv > static_cast<size_t>(INT_MAX) ||
        nhead % nkvhead != 0 || total_len < seqlen) {
        throw std::invalid_argument("invalid CUDA attention dimensions");
    }
    const size_t repeated_key_count = checkedMultiply(
        checkedMultiply(nhead, total_len, "CUDA attention repeated key"),
        d, "CUDA attention repeated key");
    const size_t repeated_value_count = checkedMultiply(
        checkedMultiply(nhead, total_len, "CUDA attention repeated value"),
        dv, "CUDA attention repeated value");
    const size_t probability_count = checkedMultiply(
        checkedMultiply(nhead, seqlen, "CUDA attention probabilities"),
        total_len, "CUDA attention probabilities");
    const size_t head_output_count = checkedMultiply(
        checkedMultiply(nhead, seqlen, "CUDA attention head output"),
        dv, "CUDA attention head output");
    size_t workspace_count = checkedAdd(
        repeated_key_count, repeated_value_count, "CUDA attention workspace");
    workspace_count = checkedAdd(
        workspace_count, probability_count, "CUDA attention workspace");
    workspace_count = checkedAdd(
        workspace_count, head_output_count, "CUDA attention workspace");
    const size_t workspace_bytes = checkedMultiply(
        workspace_count, sizeof(T), "CUDA attention workspace");
    T *workspace = nullptr;
    checkCuda(
        cudaMallocAsync(
            reinterpret_cast<void **>(&workspace), workspace_bytes, stream),
        "cudaMallocAsync attention workspace");
    T *repeated_key = workspace;
    T *repeated_value = repeated_key + repeated_key_count;
    T *probabilities = repeated_value + repeated_value_count;
    T *head_output = probabilities + probability_count;
    llaisys_cuda_repeat_kv_heads_kernel<<<
        blocks(repeated_key_count), BLOCK, 0, stream>>>(
        repeated_key, reinterpret_cast<const T *>(k), total_len,
        nhead, nkvhead, d);
    llaisys_cuda_repeat_kv_heads_kernel<<<
        blocks(repeated_value_count), BLOCK, 0, stream>>>(
        repeated_value, reinterpret_cast<const T *>(v), total_len,
        nhead, nkvhead, dv);
    checkCuda(cudaPeekAtLastError(), "CUDA attention repeat KV launch");
    cublasHandle_t handle = nullptr;
    try {
        checkCublas(cublasCreate(&handle), "cublasCreate attention");
        checkCublas(cublasSetStream(handle, stream), "cublasSetStream attention");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const auto data_type = cudaDataType<T>();
        checkCublas(
            cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(total_len), static_cast<int>(seqlen),
                static_cast<int>(d), &alpha,
                repeated_key, data_type, static_cast<int>(d),
                static_cast<long long>(total_len * d),
                reinterpret_cast<const T *>(q), data_type,
                static_cast<int>(nhead * d), static_cast<long long>(d),
                &beta, probabilities, data_type,
                static_cast<int>(total_len),
                static_cast<long long>(seqlen * total_len),
                static_cast<int>(nhead), CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmStridedBatchedEx attention QK");
        for (size_t head = 0; head < nhead; ++head) {
            launchAttentionSoftmax(
                probabilities + head * seqlen * total_len,
                seqlen, total_len, scale, stream);
        }
        checkCuda(cudaPeekAtLastError(), "CUDA attention softmax launch");
        checkCublas(
            cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(dv), static_cast<int>(seqlen),
                static_cast<int>(total_len), &alpha,
                repeated_value, data_type, static_cast<int>(dv),
                static_cast<long long>(total_len * dv),
                probabilities, data_type, static_cast<int>(total_len),
                static_cast<long long>(seqlen * total_len),
                &beta, head_output, data_type, static_cast<int>(dv),
                static_cast<long long>(seqlen * dv),
                static_cast<int>(nhead), CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmStridedBatchedEx attention PV");
        llaisys_cuda_attention_head_to_token_kernel<<<
            blocks(count), BLOCK, 0, stream>>>(
            reinterpret_cast<T *>(out), head_output, seqlen, nhead, dv);
        checkCuda(cudaPeekAtLastError(), "CUDA attention transpose launch");
        checkCublas(cublasDestroy(handle), "cublasDestroy attention");
        handle = nullptr;
        checkCuda(cudaFreeAsync(workspace, stream),
                  "cudaFreeAsync attention workspace");
    } catch (...) {
        if (handle != nullptr) cublasDestroy(handle);
        if (workspace != nullptr) cudaFreeAsync(workspace, stream);
        throw;
    }
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
