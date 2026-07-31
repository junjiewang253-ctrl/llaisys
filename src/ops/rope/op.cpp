#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 保证 out/in/pos_ids 在同一设备上（避免 CPU 读 GPU 指针等错误）
    CHECK_SAME_DEVICE(out, in, pos_ids);

    // out 和 in 必须同 shape、同 dtype（RoPE 是逐元素变换，但会跨最后一维成对旋转）
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    // 当前实现只支持连续内存（CPU 内核按线性数组访问）
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: out/in/pos_ids must be contiguous.");

    // pos_ids 必须是 int64（与 PyTorch 默认一致）
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids dtype must be int64.");

    // 检查维度：in/out 必须是 3D：[seqlen, nhead, d]
    const auto &s = in->shape();
    ASSERT(s.size() == 3, "RoPE: in/out must be 3D [seqlen, nhead, d].");
    const size_t seqlen = s[0];
    const size_t nhead = s[1];
    const size_t d = s[2];

    // d 必须为偶数，才能拆成 [a, b] 两半做旋转
    ASSERT(d % 2 == 0, "RoPE: last dim d must be even.");

    // pos_ids 形状必须是 [seqlen]
    ASSERT(pos_ids->shape().size() == 1 && pos_ids->shape()[0] == seqlen,
           "RoPE: pos_ids shape must be [seqlen].");

    // CPU fast path：如果目标设备是 CPU，直接调用 CPU 内核并返回
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         out->dtype(), seqlen, nhead, d, theta);
    }

    // 切换当前线程的运行时上下文到目标设备（例如设置 CUDA device）
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    // 分发到对应设备实现
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         out->dtype(), seqlen, nhead, d, theta);
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
