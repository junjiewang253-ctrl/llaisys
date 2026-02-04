#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // bias 允许为空：只对非空的 tensor 做设备一致性检查
    if (bias) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
    }

    // 维度约束（暂不考虑广播）
    ASSERT(out->shape().size() == 2, "Linear : out 必须是 2D 张量。");
    ASSERT(in->shape().size() == 2, "Linear: in 必须是 2D 张量。");
    ASSERT(weight->shape().size() == 2, "Linear: weight 必须是 2D 张量。");
    if (bias) {
        ASSERT(bias->shape().size() == 1, "Linear: bias 必须是 1D 张量。");
    }

    // 读取形状：in = [M, K]
    const size_t M = static_cast<size_t>(in->shape()[0]);
    const size_t K = static_cast<size_t>(in->shape()[1]);

    // weight = [N, K]（注意：计算用 W^T，所以 weight 的第二维必须等于 K）
    const size_t N = static_cast<size_t>(weight->shape()[0]);
    const size_t K2 = static_cast<size_t>(weight->shape()[1]);
    ASSERT(K2 == K, "Linear: weight.shape[1] 必须等于 in.shape[1]。");

    // out = [M, N]
    ASSERT(static_cast<size_t>(out->shape()[0]) == M, "Linear: out.shape[0] 必须等于 in.shape[0]。");
    ASSERT(static_cast<size_t>(out->shape()[1]) == N, "Linear: out.shape[1] 必须等于 weight.shape[0]。");

    // bias = [N]（若提供）
    const bool has_bias = (bias != nullptr);
    if (has_bias) {
        ASSERT(static_cast<size_t>(bias->shape()[0]) == N, "Linear: bias.shape[0] 必须等于 weight.shape[0]。");
    }

    // dtype 约束：out/in/weight 必须一致；bias（若存在）也必须一致
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (has_bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    // contiguous 约束：CPU 内核按 row-major 的线性地址计算（m*K、n*K、m*N）
    if (has_bias) {
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(),
               "Linear: 所有输入/输出必须是 contiguous。");
    } else {
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
               "Linear: out/in/weight 必须是 contiguous。");
    }

    // CPU fast path：CPU 直接计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(),
                           in->data(),
                           weight->data(),
                           has_bias ? bias->data() : nullptr,
                           out->dtype(),
                           M, K, N,
                           has_bias);
    }

    // 其他设备：先切换 context 再分发（暂不实现）
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(),
                           in->data(),
                           weight->data(),
                           has_bias ? bias->data() : nullptr,
                           out->dtype(),
                           M, K, N,
                           has_bias);
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
