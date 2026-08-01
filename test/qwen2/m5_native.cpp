#include "llaisys/error.h"
#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace {

llaisysTensor_t tensor(
    std::initializer_list<std::size_t> dimensions, float offset) {
    std::vector<std::size_t> shape(dimensions);
    auto result = tensorCreate(
        shape.data(), shape.size(), LLAISYS_DTYPE_F32,
        LLAISYS_DEVICE_CPU, 0);
    assert(result != nullptr);
    std::size_t elements = 1;
    for (auto dimension : shape) {
        elements *= dimension;
    }
    std::vector<float> values(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        const auto centered = static_cast<int>(index % 17) - 8;
        values[index] = offset + static_cast<float>(centered) * 0.0078125F;
    }
    tensorLoad(result, values.data());
    assert(llaisysGetLastErrorCode() == LLAISYS_STATUS_SUCCESS);
    return result;
}

llaisysTensor_t tokens(
    const std::vector<std::int64_t> &values,
    std::initializer_list<std::size_t> shape = {}) {
    std::vector<std::size_t> dimensions(shape);
    if (dimensions.empty()) {
        dimensions.push_back(values.size());
    }
    auto result = tensorCreate(
        dimensions.data(), dimensions.size(), LLAISYS_DTYPE_I64,
        LLAISYS_DEVICE_CPU, 0);
    assert(result != nullptr);
    if (!values.empty()) {
        tensorLoad(result, values.data());
        assert(llaisysGetLastErrorCode() == LLAISYS_STATUS_SUCCESS);
    }
    return result;
}

LlaisysQwen2Meta tiny_meta() {
    return LlaisysQwen2Meta{
        LLAISYS_DTYPE_F32,
        2, 16, 4, 2, 4, 32, 16, 32,
        1.0e-5F, 10000.0F, 2};
}

void load_weights(LlaisysQwen2Model *model) {
    auto *weights = llaisysQwen2ModelWeights(model);
    assert(weights != nullptr);
    weights->in_embed = tensor({32, 16}, 0.01F);
    weights->out_embed = tensor({32, 16}, -0.02F);
    weights->out_norm_w = tensor({16}, 1.0F);
    for (std::size_t layer = 0; layer < 2; ++layer) {
        const float delta = static_cast<float>(layer) * 0.001F;
        weights->attn_norm_w[layer] = tensor({16}, 1.0F + delta);
        weights->attn_q_w[layer] = tensor({16, 16}, 0.01F + delta);
        weights->attn_q_b[layer] = tensor({16}, 0.001F + delta);
        weights->attn_k_w[layer] = tensor({8, 16}, -0.01F + delta);
        weights->attn_k_b[layer] = tensor({8}, -0.001F + delta);
        weights->attn_v_w[layer] = tensor({8, 16}, 0.02F + delta);
        weights->attn_v_b[layer] = tensor({8}, 0.002F + delta);
        weights->attn_o_w[layer] = tensor({16, 16}, -0.02F + delta);
        weights->mlp_norm_w[layer] = tensor({16}, 1.0F - delta);
        weights->mlp_gate_w[layer] = tensor({32, 16}, 0.015F + delta);
        weights->mlp_up_w[layer] = tensor({32, 16}, -0.015F + delta);
        weights->mlp_down_w[layer] = tensor({16, 32}, 0.005F + delta);
    }
}

void cache_lifecycle() {
    auto meta = tiny_meta();
    auto *model = llaisysQwen2ModelCreate(
        &meta, LLAISYS_DEVICE_CPU, nullptr, 0);
    assert(model != nullptr);
    load_weights(model);
    assert(llaisysQwen2ModelCacheCursor(model) == 0);
    assert(llaisysQwen2ModelCacheCapacity(model) == 16);
    assert(llaisysQwen2ModelCacheAllocatedBytes(model) == 2048);
    std::vector<std::uintptr_t> addresses;
    for (std::size_t layer = 0; layer < 2; ++layer) {
        addresses.push_back(llaisysQwen2ModelCacheAddress(model, layer, 0));
        addresses.push_back(llaisysQwen2ModelCacheAddress(model, layer, 1));
    }

    const std::vector<std::int64_t> sequence{
        1, 5, 7, 9, 3, 8, 13, 2, 6, 4, 11, 10, 12, 14, 15, 16};
    for (auto token : sequence) {
        auto input = tokens({token});
        assert(llaisysQwen2ModelInferCached(model, input) >= 0);
        tensorDestroy(input);
    }
    assert(llaisysQwen2ModelCacheCursor(model) == 16);
    auto overflow = tokens({1});
    assert(llaisysQwen2ModelInferCached(model, overflow) < 0);
    tensorDestroy(overflow);
    assert(llaisysQwen2ModelCacheCursor(model) == 16);

    assert(llaisysQwen2ModelResetCache(model) == 0);
    assert(llaisysQwen2ModelCacheCursor(model) == 0);
    for (std::size_t layer = 0; layer < 2; ++layer) {
        assert(addresses[layer * 2] ==
               llaisysQwen2ModelCacheAddress(model, layer, 0));
        assert(addresses[layer * 2 + 1] ==
               llaisysQwen2ModelCacheAddress(model, layer, 1));
    }

    auto zero = tokens({});
    assert(llaisysQwen2ModelInferCached(model, zero) < 0);
    tensorDestroy(zero);
    assert(llaisysQwen2ModelCacheCursor(model) == 0);
    auto wrong_shape = tokens({1, 2}, {1, 2});
    assert(llaisysQwen2ModelInferCached(model, wrong_shape) < 0);
    tensorDestroy(wrong_shape);
    assert(llaisysQwen2ModelCacheCursor(model) == 0);
    auto wrong_dtype = tensor({1}, 1.0F);
    assert(llaisysQwen2ModelInferCached(model, wrong_dtype) < 0);
    tensorDestroy(wrong_dtype);
    assert(llaisysQwen2ModelCacheCursor(model) == 0);

    auto full = tokens({1, 5, 7, 9});
    assert(llaisysQwen2ModelInferCached(model, full) >= 0);
    assert(llaisysQwen2ModelCacheCursor(model) == 4);
    std::int64_t no_cache_tokens[]{1, 5};
    assert(llaisysQwen2ModelInfer(model, no_cache_tokens, 2) >= 0);
    assert(llaisysQwen2ModelCacheCursor(model) == 4);
    tensorDestroy(full);
    llaisysQwen2ModelDestroy(model);
}

} // namespace

int main() {
    assert(llaisysQwen2ModelResetCache(nullptr) < 0);
    assert(llaisysQwen2ModelCacheCursor(nullptr) == 0);
    assert(llaisysQwen2ModelCacheCapacity(nullptr) == 0);
    assert(llaisysQwen2ModelCacheAllocatedBytes(nullptr) == 0);
    assert(llaisysQwen2ModelCacheAddress(nullptr, 0, 0) == 0);
    for (int iteration = 0; iteration < 8; ++iteration) {
        cache_lifecycle();
    }
    return 0;
}
