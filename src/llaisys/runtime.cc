#include "llaisys/runtime.h"
#include "error.hpp"
#include "../core/context/context.hpp"
#include "../device/runtime_api.hpp"

// Llaisys API for setting context runtime.
__C void llaisysSetContextRuntime(llaisysDeviceType_t device_type, int device_id) {
    llaisys::capi::guardVoid([&] {
        llaisys::core::context().setDevice(device_type, device_id);
    });
}

// Llaisys API for getting the runtime APIs
__C const LlaisysRuntimeAPI *llaisysGetRuntimeAPI(llaisysDeviceType_t device_type) {
    return llaisys::capi::guard<const LlaisysRuntimeAPI *>(nullptr, [&] {
        return llaisys::device::getRuntimeAPI(device_type);
    });
}
