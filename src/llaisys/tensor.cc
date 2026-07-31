#include "llaisys_tensor.hpp"
#include "error.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

LlaisysTensor &requireTensor(llaisysTensor_t tensor) {
    if (tensor == nullptr || !tensor->tensor) {
        throw std::invalid_argument("tensor handle is null");
    }
    return *tensor;
}

std::vector<size_t> copySizes(const size_t *values,
                              size_t count,
                              const char *name) {
    if (count != 0 && values == nullptr) {
        throw std::invalid_argument(std::string(name) + " is null");
    }
    if (count == 0) {
        return {};
    }
    return {values, values + count};
}

} // namespace

__C {

llaisysTensor_t tensorCreate(size_t *shape,
                             size_t ndim,
                             llaisysDataType_t dtype,
                             llaisysDeviceType_t device_type,
                             int device_id) {
    return llaisys::capi::guard<llaisysTensor_t>(nullptr, [&] {
        auto shape_vec = copySizes(shape, ndim, "shape");
        return new LlaisysTensor{
            llaisys::Tensor::create(shape_vec, dtype, device_type, device_id)};
    });
}

void tensorDestroy(llaisysTensor_t tensor) {
    llaisys::capi::guardVoid([&] { delete tensor; });
}

void *tensorGetData(llaisysTensor_t tensor) {
    return llaisys::capi::guard<void *>(nullptr, [&] {
        return static_cast<void *>(requireTensor(tensor).tensor->data());
    });
}

size_t tensorGetNdim(llaisysTensor_t tensor) {
    return llaisys::capi::guard<size_t>(
        std::numeric_limits<size_t>::max(),
        [&] { return requireTensor(tensor).tensor->ndim(); });
}

void tensorGetShape(llaisysTensor_t tensor, size_t *shape) {
    llaisys::capi::guardVoid([&] {
        const auto &value = requireTensor(tensor).tensor->shape();
        if (!value.empty() && shape == nullptr) {
            throw std::invalid_argument("shape output is null");
        }
        std::copy(value.begin(), value.end(), shape);
    });
}

void tensorGetStrides(llaisysTensor_t tensor, ptrdiff_t *strides) {
    llaisys::capi::guardVoid([&] {
        const auto &value = requireTensor(tensor).tensor->strides();
        if (!value.empty() && strides == nullptr) {
            throw std::invalid_argument("strides output is null");
        }
        std::copy(value.begin(), value.end(), strides);
    });
}

llaisysDataType_t tensorGetDataType(llaisysTensor_t tensor) {
    return llaisys::capi::guard<llaisysDataType_t>(
        LLAISYS_DTYPE_INVALID,
        [&] { return requireTensor(tensor).tensor->dtype(); });
}

llaisysDeviceType_t tensorGetDeviceType(llaisysTensor_t tensor) {
    return llaisys::capi::guard<llaisysDeviceType_t>(
        LLAISYS_DEVICE_TYPE_COUNT,
        [&] { return requireTensor(tensor).tensor->deviceType(); });
}

int tensorGetDeviceId(llaisysTensor_t tensor) {
    return llaisys::capi::guard<int>(
        -1, [&] { return requireTensor(tensor).tensor->deviceId(); });
}

void tensorDebug(llaisysTensor_t tensor) {
    llaisys::capi::guardVoid([&] { requireTensor(tensor).tensor->debug(); });
}

uint8_t tensorIsContiguous(llaisysTensor_t tensor) {
    return llaisys::capi::guard<uint8_t>(
        0,
        [&] {
            return static_cast<uint8_t>(
                requireTensor(tensor).tensor->isContiguous());
        });
}

void tensorLoad(llaisysTensor_t tensor, const void *data) {
    llaisys::capi::guardVoid(
        [&] { requireTensor(tensor).tensor->load(data); });
}

llaisysTensor_t tensorView(llaisysTensor_t tensor,
                           size_t *shape,
                           size_t ndim) {
    return llaisys::capi::guard<llaisysTensor_t>(nullptr, [&] {
        auto shape_vec = copySizes(shape, ndim, "shape");
        return new LlaisysTensor{
            requireTensor(tensor).tensor->view(shape_vec)};
    });
}

llaisysTensor_t tensorPermute(llaisysTensor_t tensor, size_t *order) {
    return llaisys::capi::guard<llaisysTensor_t>(nullptr, [&] {
        auto &value = requireTensor(tensor);
        auto order_vec = copySizes(order, value.tensor->ndim(), "order");
        return new LlaisysTensor{value.tensor->permute(order_vec)};
    });
}

llaisysTensor_t tensorSlice(llaisysTensor_t tensor,
                            size_t dim,
                            size_t start,
                            size_t end) {
    return llaisys::capi::guard<llaisysTensor_t>(nullptr, [&] {
        return new LlaisysTensor{
            requireTensor(tensor).tensor->slice(dim, start, end)};
    });
}

llaisysTensor_t tensorContiguous(llaisysTensor_t tensor) {
    return llaisys::capi::guard<llaisysTensor_t>(nullptr, [&] {
        return new LlaisysTensor{
            requireTensor(tensor).tensor->contiguous()};
    });
}

llaisysTensor_t tensorReshape(llaisysTensor_t tensor,
                              size_t *shape,
                              size_t ndim) {
    return llaisys::capi::guard<llaisysTensor_t>(nullptr, [&] {
        auto shape_vec = copySizes(shape, ndim, "shape");
        return new LlaisysTensor{
            requireTensor(tensor).tensor->reshape(shape_vec)};
    });
}

llaisysTensor_t tensorTo(llaisysTensor_t tensor,
                         llaisysDeviceType_t device_type,
                         int device_id) {
    return llaisys::capi::guard<llaisysTensor_t>(nullptr, [&] {
        return new LlaisysTensor{
            requireTensor(tensor).tensor->to(device_type, device_id)};
    });
}

}
