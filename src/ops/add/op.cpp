#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/add_cpu.hpp" // 引入CPU后端实现

namespace llaisys::ops {
void add(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b); // 保证 c/a/b 在同一种设备上（比如都在 CPU，或都在 NVIDIA GPU）,否则就会出现“在 CPU 上读 GPU 指针”这种严重错误
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape()); // 当前实现只支持 完全同 shape 的逐元素加法（不做 broadcasting）
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype()); // dtype 必须一致：比如都 F32，或都 F16/BF16, 
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(), "Add: all tensors must be contiguous."); // 只支持连续内存（contiguous）。因为 CPU 内核按线性数组访问：for i in [0, numel)

    // always support cpu calculation
    if (c->deviceType() == LLAISYS_DEVICE_CPU) { // 这里相当于一个 fast path：如果是 CPU，直接调用 CPU 内核并返回
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    }
    // Context 记录当前激活设备
    llaisys::core::context().setDevice(c->deviceType(), c->deviceId()); // 在真正调用设备实现之前，先把当前线程的运行时上下文切到目标设备（例如设置 CUDA device）
    // switch 分发到对应设备实现
    switch (c->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
// NVIDIA 分支被宏保护：没开 CUDA 的时候编译器根本看不到这段
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
