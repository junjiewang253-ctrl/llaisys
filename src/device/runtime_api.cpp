#include "runtime_api.hpp"
#include "../llaisys/error.hpp"

namespace llaisys::device {

int getDeviceCount() {
    return llaisys::capi::guard<int>(-1, [] {
        throw llaisys::capi::NotSupportedError("unsupported device runtime");
        return 0;
    });
}

void setDevice(int) {
    llaisys::capi::guardVoid([] {
        throw llaisys::capi::NotSupportedError("unsupported device runtime");
    });
}

void deviceSynchronize() {
    llaisys::capi::guardVoid([] {
        throw llaisys::capi::NotSupportedError("unsupported device runtime");
    });
}

llaisysStream_t createStream() {
    return llaisys::capi::guard<llaisysStream_t>(nullptr, [] {
        throw llaisys::capi::NotSupportedError("unsupported device runtime");
        return static_cast<llaisysStream_t>(nullptr);
    });
}

void destroyStream(llaisysStream_t) {
    llaisys::capi::guardVoid([] {
        throw llaisys::capi::NotSupportedError("unsupported device runtime");
    });
}
void streamSynchronize(llaisysStream_t) {
    llaisys::capi::guardVoid([] {
        throw llaisys::capi::NotSupportedError("unsupported device runtime");
    });
}

void *mallocDevice(size_t) {
    return llaisys::capi::guard<void *>(nullptr, [] {
        throw llaisys::capi::NotSupportedError("unsupported device runtime");
        return static_cast<void *>(nullptr);
    });
}

void freeDevice(void *) {
    llaisys::capi::guardVoid([] {
        throw llaisys::capi::NotSupportedError("unsupported device runtime");
    });
}

void *mallocHost(size_t) {
    return llaisys::capi::guard<void *>(nullptr, [] {
        throw llaisys::capi::NotSupportedError("unsupported device runtime");
        return static_cast<void *>(nullptr);
    });
}

void freeHost(void *) {
    llaisys::capi::guardVoid([] {
        throw llaisys::capi::NotSupportedError("unsupported device runtime");
    });
}

void memcpySync(void *, const void *, size_t, llaisysMemcpyKind_t) {
    llaisys::capi::guardVoid([] {
        throw llaisys::capi::NotSupportedError("unsupported device runtime");
    });
}

void memcpyAsync(void *, const void *, size_t, llaisysMemcpyKind_t, llaisysStream_t) {
    llaisys::capi::guardVoid([] {
        throw llaisys::capi::NotSupportedError("unsupported device runtime");
    });
}

static const LlaisysRuntimeAPI NOOP_RUNTIME_API = {
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

const LlaisysRuntimeAPI *getUnsupportedRuntimeAPI() {
    return &NOOP_RUNTIME_API;
}

const LlaisysRuntimeAPI *getRuntimeAPI(llaisysDeviceType_t device_type) {
    // Implement for all device types
    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return llaisys::device::cpu::getRuntimeAPI();
    case LLAISYS_DEVICE_NVIDIA:
#ifdef ENABLE_NVIDIA_API
        return llaisys::device::nvidia::getRuntimeAPI();
#else
        return getUnsupportedRuntimeAPI();
#endif
    default:
        throw std::invalid_argument("invalid device type");
    }
}
} // namespace llaisys::device
