#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>
#include <cstdint>

namespace llaisys::ops::cpu {

// 返回某个 dtype 的元素字节数（用于按字节 memcpy）
// 注意：Embedding 本质是“按行拷贝”，不需要做数值计算，因此只要知道每个元素占多少字节即可。
static inline size_t dtype_size(llaisysDataType_t t) {
	switch(t) {
    case LLAISYS_DTYPE_F32: return 4;
    case LLAISYS_DTYPE_F16: return 2;
    case LLAISYS_DTYPE_BF16: return 2;
    default:
        // 遇到不支持的 dtype（比如 int8 / int32 等）直接报错
        EXCEPTION_UNSUPPORTED_DATATYPE(t);
	}
}

void embedding(std::byte* out,
               const std::byte* index,
               const std::byte* weight,
               llaisysDataType_t out_type,
               llaisysDataType_t index_type,
               llaisysDataType_t weight_type,
               size_t N, // index 长度
               size_t D, // embedding 维度（每行长度）
               size_t V) {
    // 按题目要求：index 必须是 int64
    if (index_type != LLAISYS_DTYPE_I64) {
        EXCEPTION_UNSUPPORTED_DATATYPE(index_type);
    }

    // 这里约束 out 和 weight dtype 必须一致（和很多框架默认一致）
    // 这样我们可以直接 memcpy 行数据（无需 dtype 转换）
    if (out_type != weight_type) {
        ASSERT(false, "Embedding: out dtype must equal weight dtype.");
    }

    const size_t elem_bytes = dtype_size(out_type); // 每个元素的字节数（f32=4, f16/bf16=2）
    const size_t row_bytes = D * elem_bytes; // 每一行 embedding 的总字节数

    // index 是 int64，直接把 byte* 转成 int64_t* 来读索引
    const auto *idx = reinterpret_cast<const int64_t *>(index);

    // 对每个index[i], 从weight的第r行拷贝到out的第i行
    for (size_t i = 0; i < N; ++i) {
        const int64_t r = idx[i];
        // 越界检查：必须满足0<=r<V
        ASSERT(r >= 0 && static_cast<size_t>(r) < V, "Embedding: index out of range.");

        // src = source = 源数据地址（从 weight 的第 r 行开始）, dst = destination = 目标地址（写到 out 的第 i 行）
        // 二维矩阵按行连续存储（row-major）时, 第 k 行的起始地址 = base + k * 每行字节数, 每行字节数 = D * 每个元素字节数
        const std::byte *src = weight + static_cast<size_t>(r) * row_bytes; // weight[r, :], weight + r * row_bytes 就是 weight[r, 0] 那个元素的地址（这一整行的起点）
        std::byte *dst = out + i * row_bytes; // out + i * row_bytes 就是 out[i, 0] 的地址

        // 直接拷贝整行，效率高
        // memcpy 来自头文件 <cstring>，用于按字节拷贝一段连续内存。
        // 语法：void* memcpy(void* dest, const void* src, size_t count);
        // dest：目标地址（写到哪里）,src：源地址（从哪里读）, count：要拷贝的字节数
        std::memcpy(dst, src, row_bytes);
    }
}
}