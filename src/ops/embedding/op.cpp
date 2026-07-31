#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp" // 引入CPU后端实现

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 设备一致性：避免拿 CPU 指针去读 GPU 内存等严重问题
    CHECK_SAME_DEVICE(out, index, weight);

    // 形状约束：
    // weight: [V, D]  二维
    // index:  [N]     一维
    // out:    [N, D]  二维
    ASSERT(weight->shape().size() == 2, "Embedding: weight must be 2-D.");
    ASSERT(index->shape().size() == 1, "Embedding: index must be 1-D.");
    ASSERT(out->shape().size() == 2, "Embedding: out must be 2-D.");

    const size_t V = static_cast<size_t>(weight->shape()[0]); // 词表大小（行数）
    const size_t D = static_cast<size_t>(weight->shape()[1]); // embedding维度
    const size_t N = static_cast<size_t>(index->shape()[0]); // index长度

    // out形状必须匹配
    ASSERT(static_cast<size_t>(out->shape()[0]) == N, "Embedding: out.shape[0] must equal index.shape[0].");
    ASSERT(static_cast<size_t>(out->shape()[1]) == D, "Embedding: out.shape[1] must equal weight.shape[1].");

    // dtype 约束：index 必须 int64；out 与 weight dtype 必须一致（f32/f16/bf16）
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index dtype must be int64.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    // 只支持 contiguous：因为 CPU 内核按线性内存 memcpy
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");

    // CPU fast path：如果是 CPU，直接调用 CPU 内核
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), 
                              index->data(),
                              weight->data(), out->dtype(), index->dtype(), weight->dtype(), N, D, V);
    }
    // 其他设备：切换 context + 分发（暂时不实现）
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(),
                              index->data(),
                              weight->data(),
                              out->dtype(),
                              index->dtype(),
                              weight->dtype(),
                              N, D, V);
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
