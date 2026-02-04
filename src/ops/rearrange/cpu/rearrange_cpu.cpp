#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

// 递归/迭代都可以，这里用“多维索引 + 线性推进”的通用实现
template <typename T>
static void rearrange_strided_impl(
    T *out,
    const T *in,
    const size_t *shape,
    const ptrdiff_t *out_strides,
    const ptrdiff_t *in_strides,
    size_t ndim) {

    if (ndim == 0) {
        return;
    }

    // 维护一个 ndim 维的 index
    std::vector<size_t> idx(ndim, 0);

    while (true) {
        // 计算当前元素的 in/out 偏移（以元素为单位）
        ptrdiff_t o_off = 0;
        ptrdiff_t i_off = 0;
        for (size_t d = 0; d < ndim; ++d) {
            o_off += static_cast<ptrdiff_t>(idx[d]) * out_strides[d];
            i_off += static_cast<ptrdiff_t>(idx[d]) * in_strides[d];
        }

        out[o_off] = in[i_off];

        // idx ++（类似 odometer）
        size_t d = ndim;
        while (d > 0) {
            --d;
            idx[d]++;
            if (idx[d] < shape[d]) {
                break;
            }
            idx[d] = 0;
            if (d == 0) {
                return; // 全部遍历结束
            }
        }
    }
}

namespace llaisys::ops::cpu {

void rearrange(
    std::byte *out_data,
    const std::byte *in_data,
    llaisysDataType_t dtype,
    const size_t *shape,
    const ptrdiff_t *out_strides,
    const ptrdiff_t *in_strides,
    size_t ndim) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rearrange_strided_impl(
            reinterpret_cast<float *>(out_data),
            reinterpret_cast<const float *>(in_data),
            shape, out_strides, in_strides, ndim);

    case LLAISYS_DTYPE_F16:
        return rearrange_strided_impl(
            reinterpret_cast<llaisys::fp16_t *>(out_data),
            reinterpret_cast<const llaisys::fp16_t *>(in_data),
            shape, out_strides, in_strides, ndim);

    case LLAISYS_DTYPE_BF16:
        return rearrange_strided_impl(
            reinterpret_cast<llaisys::bf16_t *>(out_data),
            reinterpret_cast<const llaisys::bf16_t *>(in_data),
            shape, out_strides, in_strides, ndim);

    case LLAISYS_DTYPE_I64:
        return rearrange_strided_impl(
            reinterpret_cast<int64_t *>(out_data),
            reinterpret_cast<const int64_t *>(in_data),
            shape, out_strides, in_strides, ndim);

    // 你也可以按需补更多类型：I32/U8 等
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu