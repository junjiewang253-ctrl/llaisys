#include "op.hpp"
#pragma message("Compiling NEW argmax/op.cpp")

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    std::fprintf(stderr, "[llaisys] entered C++ ops::argmax device=%d dtype=%d numel=%zu\n",
                 (int)vals->deviceType(), (int)vals->dtype(), (size_t)vals->numel());
    // 1) 设备一致性检查：三个张量必须在同一设备上（比如都在 CPU）
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    // 2) 形状约束（作业阶段先简化）
    // vals 必须是 1D
    ASSERT(vals->shape().size() == 1, "Argmax: vals 目前必须是 1D 张量。");
    // max_idx 必须是 1D 且只有一个元素（保持维度：用 [1] 表示标量）
    ASSERT(max_idx->shape().size() == 1 && max_idx->numel() == 1, "Argmax: max_idx 必须是 1D 且只有 1 个元素。");
    // max_val 必须是 1D 且只有一个元素
    ASSERT(max_val->shape().size() == 1 && max_val->numel() == 1, "Argmax: max_val 必须是 1D 且只有 1 个元素。");
    // 3) dtype 约束
    // max_val 的 dtype 必须与 vals 相同（最大值用同类型保存）
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    // max_idx 必须是 int64（PyTorch 的默认 index dtype）
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx 的 dtype 必须是 int64。");
    // 4) contiguous 约束：当前实现只支持连续内存（方便按线性数组遍历）
    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(), "Argmax: 当前要求所有张量必须是 contiguous。");
    // 5) 空输入处理：空张量没有定义 argmax，这里直接报错
    ASSERT(vals->numel() > 0, "Argmax: vals 不能为空。");
    // 6) CPU 快速路径：如果在 CPU 上，直接调用 CPU 后端实现
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }
    // 7) 非 CPU 情况：切换当前线程 Context 的激活设备
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    // 8) 按设备分发到不同后端实现
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
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
