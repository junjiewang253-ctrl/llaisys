#include "runtime.hpp"

#include "../../device/runtime_api.hpp"
#include <new>

namespace llaisys::core {
Runtime::Runtime(llaisysDeviceType_t device_type, int device_id)
    : _device_type(device_type), _device_id(device_id), _is_active(false) {
    _api = llaisys::device::getRuntimeAPI(_device_type);
    _stream = _api->create_stream();
}

Runtime::~Runtime() {
    _api->destroy_stream(_stream);
    _api = nullptr;
}

void Runtime::_activate() {
    _api->set_device(_device_id);
    _is_active = true;
}

void Runtime::_deactivate() {
    _is_active = false;
}

bool Runtime::isActive() const {
    return _is_active;
}

llaisysDeviceType_t Runtime::deviceType() const {
    return _device_type;
}

int Runtime::deviceId() const {
    return _device_id;
}

const LlaisysRuntimeAPI *Runtime::api() const {
    return _api;
}

storage_t Runtime::allocateDeviceStorage(size_t size) {
    auto memory = static_cast<std::byte *>(_api->malloc_device(size));
    if (size != 0 && memory == nullptr) {
        throw std::bad_alloc();
    }
    return std::shared_ptr<Storage>(
        new Storage(memory, size, _api, _device_type, _device_id, false));
}

storage_t Runtime::allocateHostStorage(size_t size) {
    auto memory = static_cast<std::byte *>(_api->malloc_host(size));
    if (size != 0 && memory == nullptr) {
        throw std::bad_alloc();
    }
    return std::shared_ptr<Storage>(
        new Storage(memory, size, _api, LLAISYS_DEVICE_CPU, 0, true));
}

llaisysStream_t Runtime::stream() const {
    return _stream;
}

void Runtime::synchronize() const {
    _api->stream_synchronize(_stream);
}

} // namespace llaisys::core
