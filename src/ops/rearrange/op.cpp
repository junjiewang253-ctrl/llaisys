#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"

namespace llaisys::ops {

void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    // CPU 先实现；后续可加 GPU
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        const size_t ndim = out->ndim();
        ASSERT(in->ndim() == ndim, "Rearrange: ndim mismatch.");

        // 关键：取 shape/stride（stride 单位需与你 tensor 实现一致）
        // 这里假设 stride 是“以元素为单位”的 ptrdiff_t 数组
        return cpu::rearrange(
            out->data(),
            in->data(),
            out->dtype(),
            out->shape().data(),
            out->strides().data(),
            in->strides().data(),
            ndim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(
            out->data(),
            in->data(),
            out->dtype(),
            out->shape().data(),
            out->strides().data(),
            in->strides().data(),
            out->ndim());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
