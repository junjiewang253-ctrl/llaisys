#include "../runtime_api.hpp"
#include "../../llaisys/error.hpp"

#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

namespace llaisys::device::cpu {

namespace runtime_api {
int getDeviceCount() {
    return llaisys::capi::guard<int>(-1, [] { return 1; });
}

void setDevice(int device) {
    llaisys::capi::guardVoid([&] {
        if (device != 0) {
            throw std::invalid_argument("CPU device id must be 0");
        }
    });
}

void deviceSynchronize() {
    llaisys::capi::guardVoid([] {});
}

llaisysStream_t createStream() {
    return llaisys::capi::guard<llaisysStream_t>(
        nullptr, [] { return static_cast<llaisysStream_t>(nullptr); });
}

void destroyStream(llaisysStream_t) {
    llaisys::capi::guardVoid([] {});
}
void streamSynchronize(llaisysStream_t) {
    llaisys::capi::guardVoid([] {});
}

void *mallocDevice(size_t size) {
    return llaisys::capi::guard<void *>(nullptr, [&] {
        void *result = std::malloc(size == 0 ? 1 : size);
        if (result == nullptr) {
            throw std::bad_alloc();
        }
        return result;
    });
}

void freeDevice(void *ptr) {
    llaisys::capi::guardVoid([&] { std::free(ptr); });
}

void *mallocHost(size_t size) {
    return mallocDevice(size);
}

void freeHost(void *ptr) {
    freeDevice(ptr);
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    const int raw_kind = llaisys::capi::enumValue(kind);
    llaisys::capi::guardVoid([&] {
        if (raw_kind < LLAISYS_MEMCPY_H2H
            || raw_kind > LLAISYS_MEMCPY_D2D) {
            throw std::invalid_argument("invalid memcpy kind");
        }
        if (size != 0 && (dst == nullptr || src == nullptr)) {
            throw std::invalid_argument("null memcpy pointer");
        }
        if (size != 0) {
            std::memmove(dst, src, size);
        }
    });
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    (void)stream;
    memcpySync(dst, src, size, kind);
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
} // namespace llaisys::device::cpu
