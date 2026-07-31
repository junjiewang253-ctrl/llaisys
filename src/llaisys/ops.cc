#include "llaisys/ops.h"

#include "error.hpp"
#include "llaisys_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

#include <stdexcept>
#include <string>

namespace {

llaisys::tensor_t requireTensor(llaisysTensor_t tensor, const char *name) {
    if (tensor == nullptr || !tensor->tensor) {
        throw std::invalid_argument(std::string(name) + " tensor is null");
    }
    return tensor->tensor;
}

} // namespace

__C {

void llaisysAdd(llaisysTensor_t out,
                llaisysTensor_t lhs,
                llaisysTensor_t rhs) {
    llaisys::capi::guardVoid([&] {
        llaisys::ops::add(
            requireTensor(out, "out"),
            requireTensor(lhs, "lhs"),
            requireTensor(rhs, "rhs"));
    });
}

void llaisysArgmax(llaisysTensor_t max_idx,
                   llaisysTensor_t max_val,
                   llaisysTensor_t values) {
    llaisys::capi::guardVoid([&] {
        llaisys::ops::argmax(
            requireTensor(max_idx, "max_idx"),
            requireTensor(max_val, "max_val"),
            requireTensor(values, "values"));
    });
}

void llaisysEmbedding(llaisysTensor_t out,
                      llaisysTensor_t index,
                      llaisysTensor_t weight) {
    llaisys::capi::guardVoid([&] {
        llaisys::ops::embedding(
            requireTensor(out, "out"),
            requireTensor(index, "index"),
            requireTensor(weight, "weight"));
    });
}

void llaisysLinear(llaisysTensor_t out,
                   llaisysTensor_t input,
                   llaisysTensor_t weight,
                   llaisysTensor_t bias) {
    llaisys::capi::guardVoid([&] {
        llaisys::ops::linear(
            requireTensor(out, "out"),
            requireTensor(input, "input"),
            requireTensor(weight, "weight"),
            bias == nullptr ? llaisys::tensor_t{}
                            : requireTensor(bias, "bias"));
    });
}

void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t input) {
    llaisys::capi::guardVoid([&] {
        llaisys::ops::rearrange(
            requireTensor(out, "out"), requireTensor(input, "input"));
    });
}

void llaisysRmsNorm(llaisysTensor_t out,
                    llaisysTensor_t input,
                    llaisysTensor_t weight,
                    float eps) {
    llaisys::capi::guardVoid([&] {
        llaisys::ops::rms_norm(
            requireTensor(out, "out"),
            requireTensor(input, "input"),
            requireTensor(weight, "weight"),
            eps);
    });
}

void llaisysROPE(llaisysTensor_t out,
                 llaisysTensor_t input,
                 llaisysTensor_t pos_ids,
                 float theta) {
    llaisys::capi::guardVoid([&] {
        llaisys::ops::rope(
            requireTensor(out, "out"),
            requireTensor(input, "input"),
            requireTensor(pos_ids, "pos_ids"),
            theta);
    });
}

void llaisysSelfAttention(llaisysTensor_t out,
                          llaisysTensor_t q,
                          llaisysTensor_t k,
                          llaisysTensor_t v,
                          float scale) {
    llaisys::capi::guardVoid([&] {
        llaisys::ops::self_attention(
            requireTensor(out, "out"),
            requireTensor(q, "q"),
            requireTensor(k, "k"),
            requireTensor(v, "v"),
            scale);
    });
}

void llaisysSwiGLU(llaisysTensor_t out,
                   llaisysTensor_t gate,
                   llaisysTensor_t up) {
    llaisys::capi::guardVoid([&] {
        llaisys::ops::swiglu(
            requireTensor(out, "out"),
            requireTensor(gate, "gate"),
            requireTensor(up, "up"));
    });
}

}
