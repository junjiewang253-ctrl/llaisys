#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous.");

    // shape checks:
    // attn_val: [seqlen, nhead, dv]
    // q:        [seqlen, nhead, d]
    // k:        [total_len, nkvhead, d]
    // v:        [total_len, nkvhead, dv]
    ASSERT(attn_val->shape().size() == 3 && q->shape().size() == 3 && k->shape().size() == 3 && v->shape().size() == 3,
           "self_attention: all tensors must be 3D.");

    const size_t seqlen = q->shape()[0];
    const size_t nhead = q->shape()[1];
    const size_t d = q->shape()[2];

    const size_t total_len = k->shape()[0];
    const size_t nkvhead = k->shape()[1];
    const size_t d_k = k->shape()[2];

    const size_t total_len_v = v->shape()[0];
    const size_t nkvhead_v = v->shape()[1];
    const size_t dv = v->shape()[2];

    ASSERT(d_k == d, "self_attention: k last dim must equal q last dim.");
    ASSERT(total_len_v == total_len, "self_attention: v total_len must equal k total_len.");
    ASSERT(nkvhead_v == nkvhead, "self_attention: v nkvhead must equal k nkvhead.");

    ASSERT(attn_val->shape()[0] == seqlen, "self_attention: attn_val seqlen mismatch.");
    ASSERT(attn_val->shape()[1] == nhead, "self_attention: attn_val nhead mismatch.");
    ASSERT(attn_val->shape()[2] == dv, "self_attention: attn_val dv mismatch.");

    // 常见约束：nhead 是 nkvhead 的整数倍（GQA）
    ASSERT(nkvhead > 0, "self_attention: nkvhead must be > 0.");
    ASSERT(nhead % nkvhead == 0, "self_attention: require nhead % nkvhead == 0 for GQA.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(),
                                   q->data(),
                                   k->data(),
                                   v->data(),
                                   attn_val->dtype(),
                                   seqlen, nhead, nkvhead, d, dv, total_len,
                                   scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(),
                                   q->data(),
                                   k->data(),
                                   v->data(),
                                   attn_val->dtype(),
                                   seqlen, nhead, nkvhead, d, dv, total_len,
                                   scale);
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
