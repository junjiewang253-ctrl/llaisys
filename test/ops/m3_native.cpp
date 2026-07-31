#include "llaisys/error.h"
#include "llaisys/ops.h"
#include "llaisys/tensor.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

template <std::size_t N>
llaisysTensor_t tensor(std::array<std::size_t, N> shape,
                      llaisysDataType_t dtype = LLAISYS_DTYPE_F32) {
    auto result = tensorCreate(
        shape.data(), shape.size(), dtype, LLAISYS_DEVICE_CPU, 0);
    assert(result != nullptr);
    return result;
}

void success() {
    assert(llaisysGetLastErrorCode() == LLAISYS_STATUS_SUCCESS);
}

} // namespace

int main() {
    std::array<float, 4> values{1.0F, -2.0F, 3.0F, 4.0F};
    std::array<float, 4> weights{0.5F, 1.0F, -0.5F, 2.0F};
    std::array<std::int64_t, 2> indices{0, 1};

    auto vector = tensor(std::array<std::size_t, 1>{4});
    auto vector2 = tensor(std::array<std::size_t, 1>{4});
    auto vector_out = tensor(std::array<std::size_t, 1>{4});
    tensorLoad(vector, values.data());
    tensorLoad(vector2, weights.data());
    success();

    llaisysAdd(vector_out, vector, vector2);
    success();
    llaisysSwiGLU(vector_out, vector, vector2);
    success();
    llaisysRearrange(vector_out, vector);
    success();

    auto max_index =
        tensor(std::array<std::size_t, 1>{1}, LLAISYS_DTYPE_I64);
    auto max_value = tensor(std::array<std::size_t, 1>{1});
    llaisysArgmax(max_index, max_value, vector);
    success();

    auto index =
        tensor(std::array<std::size_t, 1>{2}, LLAISYS_DTYPE_I64);
    auto embedding_weight = tensor(std::array<std::size_t, 2>{2, 2});
    auto matrix_out = tensor(std::array<std::size_t, 2>{2, 2});
    tensorLoad(index, indices.data());
    tensorLoad(embedding_weight, weights.data());
    llaisysEmbedding(matrix_out, index, embedding_weight);
    success();

    auto matrix_in = tensor(std::array<std::size_t, 2>{2, 2});
    tensorLoad(matrix_in, values.data());
    llaisysLinear(matrix_out, matrix_in, embedding_weight, nullptr);
    success();
    auto norm_weight = tensor(std::array<std::size_t, 1>{2});
    tensorLoad(norm_weight, weights.data());
    llaisysRmsNorm(matrix_out, matrix_in, norm_weight, 1.0e-5F);
    success();

    auto sequence = tensor(std::array<std::size_t, 3>{1, 1, 2});
    auto sequence_out = tensor(std::array<std::size_t, 3>{1, 1, 2});
    auto position =
        tensor(std::array<std::size_t, 1>{1}, LLAISYS_DTYPE_I64);
    tensorLoad(sequence, values.data());
    tensorLoad(position, indices.data());
    llaisysROPE(sequence_out, sequence, position, 10000.0F);
    success();
    llaisysSelfAttention(
        sequence_out, sequence, sequence, sequence, 0.70710678F);
    success();

    llaisysAdd(nullptr, vector, vector2);
    assert(llaisysGetLastErrorCode() != LLAISYS_STATUS_SUCCESS);
    llaisysRmsNorm(matrix_out, matrix_in, norm_weight, 0.0F);
    assert(llaisysGetLastErrorCode() != LLAISYS_STATUS_SUCCESS);

    for (auto item : {vector, vector2, vector_out, max_index, max_value,
                      index, embedding_weight, matrix_out, matrix_in,
                      norm_weight, sequence, sequence_out, position}) {
        tensorDestroy(item);
    }
    success();
    return 0;
}
