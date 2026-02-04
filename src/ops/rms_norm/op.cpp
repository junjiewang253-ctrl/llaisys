#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 1) 设备与 dtype 检查：三者必须在同一设备 & dtype 相同
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    // 2) contiguous 检查：当前 CPU 内核按 row-major 连续内存访问
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: out/in/weight must be contiguous.");

    // 3) 形状约定（按题意先做最简版本）
    // in/out: 2D；weight: 1D；且 weight[0] == D（in 的最后一维）
    ASSERT(in->shape().size() == 2, "RMSNorm: input must be 2D for now.");
    ASSERT(out->shape().size() == 2, "RMSNorm: output must be 2D for now.");
    ASSERT(weight->shape().size() == 1, "RMSNorm: weight must be 1D for now.");

    // out shape必须等于in shape
    ASSERT(in->shape()[0] == out->shape()[0] && in->shape()[1] == out->shape()[1],
           "RMSNorm: out shape must equal in shape.");

    const size_t M = in->shape()[0];
    const size_t D = in->shape()[1];

    // weight的长度必须等于D
    ASSERT(weight->shape()[0] == D, "RMSNorm: weight length must equal input last dimension.");

    // 4) CPU 快路径：直接调用 CPU 内核
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), M, D, eps);
    }

    // 5) 其他设备：切换上下文后分发（目前 NVIDIA 未实现）
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId()); 
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), M, D, eps);
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
