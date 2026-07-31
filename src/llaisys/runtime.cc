#include "llaisys/runtime.h"
#include "error.hpp"
#include "../core/context/context.hpp"
#include "../device/runtime_api.hpp"

// Llaisys API for setting context runtime.
__C void llaisysSetContextRuntime(llaisysDeviceType_t device_type, int device_id) {
    const int raw_device_type = llaisys::capi::enumValue(device_type);
    llaisys::capi::guardVoid([&] {
        if (raw_device_type < LLAISYS_DEVICE_CPU
            || raw_device_type >= LLAISYS_DEVICE_TYPE_COUNT) {
            throw std::invalid_argument("invalid device type");
        }
        llaisys::core::context().setDevice(
            static_cast<llaisysDeviceType_t>(raw_device_type), device_id);
    });
}

// Llaisys API for getting the runtime APIs
__C const LlaisysRuntimeAPI *llaisysGetRuntimeAPI(llaisysDeviceType_t device_type) {
    const int raw_device_type = llaisys::capi::enumValue(device_type);
    return llaisys::capi::guard<const LlaisysRuntimeAPI *>(nullptr, [&] {
        if (raw_device_type < LLAISYS_DEVICE_CPU
            || raw_device_type >= LLAISYS_DEVICE_TYPE_COUNT) {
            throw std::invalid_argument("invalid device type");
        }
        return llaisys::device::getRuntimeAPI(
            static_cast<llaisysDeviceType_t>(raw_device_type));
    });
}
