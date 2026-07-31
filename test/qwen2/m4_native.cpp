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

void infer_repeatedly() {
    auto meta = tiny_meta();
    auto *model = llaisysQwen2ModelCreate(
        &meta, LLAISYS_DEVICE_CPU, nullptr, 0);
    assert(model != nullptr);
    load_weights(model);
    std::vector<std::int64_t> tokens{1, 5, 7, 9, 3, 8, 13, 2};
    for (std::size_t length : {std::size_t{1}, std::size_t{4},
                               std::size_t{8}, std::size_t{3}}) {
        const auto next =
            llaisysQwen2ModelInfer(model, tokens.data(), length);
        assert(next >= 0 && next < 32);
        const auto *trace = llaisysQwen2ModelTrace(model);
        assert(trace != nullptr);
        assert(trace->embedding != nullptr);
        assert(trace->attention_out != nullptr);
        assert(trace->layer_output != nullptr);
        assert(trace->attention_out[0] != nullptr);
        assert(trace->attention_out[1] != nullptr);
        assert(trace->layer_output[0] != nullptr);
        assert(trace->layer_output[1] != nullptr);
        assert(trace->final_norm != nullptr);
        assert(trace->logits != nullptr);
        assert(trace->greedy_token == next);
    }
    llaisysQwen2ModelDestroy(model);
}

} // namespace

int main() {
    auto invalid = tiny_meta();
    invalid.nlayer = 0;
    assert(llaisysQwen2ModelCreate(
               &invalid, LLAISYS_DEVICE_CPU, nullptr, 0) == nullptr);

    auto meta = tiny_meta();
    auto *missing = llaisysQwen2ModelCreate(
        &meta, LLAISYS_DEVICE_CPU, nullptr, 0);
    assert(missing != nullptr);
    std::int64_t token = 1;
    assert(llaisysQwen2ModelInfer(missing, &token, 1) == -2);
    llaisysQwen2ModelDestroy(missing);

    infer_repeatedly();
    infer_repeatedly();
    llaisysQwen2ModelDestroy(nullptr);
    return 0;
}
