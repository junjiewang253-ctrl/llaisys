#include "../runtime_api.hpp"
#include "../../llaisys/error.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::nvidia {

namespace runtime_api {
namespace {
void checkCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

cudaMemcpyKind copyKind(llaisysMemcpyKind_t kind) {
    switch (llaisys::capi::enumValue(kind)) {
    case LLAISYS_MEMCPY_H2H: return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D: return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H: return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D: return cudaMemcpyDeviceToDevice;
    default: throw std::invalid_argument("invalid CUDA memcpy kind");
    }
}

void validateCopy(const void *dst, const void *src, size_t size) {
    if (size != 0 && (dst == nullptr || src == nullptr)) {
        throw std::invalid_argument("null CUDA memcpy pointer");
    }
}
} // namespace

int getDeviceCount() {
    return llaisys::capi::guard<int>(-1, [] {
        int count = 0;
        checkCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        return count;
    });
}

void setDevice(int device) {
    llaisys::capi::guardVoid([&] {
        if (device < 0) {
            throw std::invalid_argument("CUDA device id must be non-negative");
        }
        const auto status = cudaSetDevice(device);
        if (status == cudaErrorInvalidDevice) {
            throw std::invalid_argument("CUDA device id is not visible");
        }
        checkCuda(status, "cudaSetDevice");
    });
}

void deviceSynchronize() {
    llaisys::capi::guardVoid(
        [] { checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); });
}

llaisysStream_t createStream() {
    return llaisys::capi::guard<llaisysStream_t>(nullptr, [] {
        cudaStream_t stream = nullptr;
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags");
        return reinterpret_cast<llaisysStream_t>(stream);
    });
}

void destroyStream(llaisysStream_t stream) {
    llaisys::capi::guardVoid([&] {
        if (stream == nullptr) {
            throw std::invalid_argument("null CUDA stream");
        }
        checkCuda(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)),
                  "cudaStreamDestroy");
    });
}
void streamSynchronize(llaisysStream_t stream) {
    llaisys::capi::guardVoid([&] {
        if (stream == nullptr) {
            throw std::invalid_argument("null CUDA stream");
        }
        checkCuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
                  "cudaStreamSynchronize");
    });
}

void *mallocDevice(size_t size) {
    return llaisys::capi::guard<void *>(nullptr, [&] {
        void *pointer = nullptr;
        checkCuda(cudaMalloc(&pointer, size == 0 ? 1 : size), "cudaMalloc");
        return pointer;
    });
}

void freeDevice(void *ptr) {
    llaisys::capi::guardVoid(
        [&] { checkCuda(cudaFree(ptr), "cudaFree"); });
}

void *mallocHost(size_t size) {
    return llaisys::capi::guard<void *>(nullptr, [&] {
        void *pointer = nullptr;
        checkCuda(cudaMallocHost(&pointer, size == 0 ? 1 : size),
                  "cudaMallocHost");
        return pointer;
    });
}

void freeHost(void *ptr) {
    llaisys::capi::guardVoid(
        [&] { checkCuda(cudaFreeHost(ptr), "cudaFreeHost"); });
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    llaisys::capi::guardVoid([&] {
        validateCopy(dst, src, size);
        const auto cuda_kind = copyKind(kind);
        if (size != 0) {
            checkCuda(cudaMemcpy(dst, src, size, cuda_kind), "cudaMemcpy");
        }
    });
}

void memcpyAsync(void *dst, const void *src, size_t size,
                 llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    llaisys::capi::guardVoid([&] {
        validateCopy(dst, src, size);
        if (stream == nullptr) {
            throw std::invalid_argument("null CUDA stream");
        }
        const auto cuda_kind = copyKind(kind);
        if (size != 0) {
            checkCuda(cudaMemcpyAsync(dst, src, size, cuda_kind,
                                      reinterpret_cast<cudaStream_t>(stream)),
                      "cudaMemcpyAsync");
        }
    });
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
