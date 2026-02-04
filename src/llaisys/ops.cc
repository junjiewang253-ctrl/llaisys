#include "llaisys/ops.h"

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

#include <cstdio>

__C {

    void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b) {
        if (!c || !a || !b) {
            std::fprintf(stderr, "[llaisys] llaisysAdd: nullptr arg(s): c=%p a=%p b=%p\n", (void *)c, (void *)a, (void *)b);
            return;
        }
        llaisys::ops::add(c->tensor, a->tensor, b->tensor);
    }

    void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals) {
        if (!max_idx || !max_val || !vals) {
            std::fprintf(stderr, "[llaisys] llaisysArgmax: nullptr arg(s): max_idx=%p max_val=%p vals=%p\n",
                         (void *)max_idx, (void *)max_val, (void *)vals);
            return;
        }
        std::fprintf(stderr, "[llaisys] reached llaisysArgmax UNIQUE=20260127_001\n");
        llaisys::ops::argmax(max_idx->tensor, max_val->tensor, vals->tensor);
    }

    void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight) {
        if (!out || !index || !weight) {
            std::fprintf(stderr, "[llaisys] llaisysEmbedding: nullptr arg(s): out=%p index=%p weight=%p\n",
                         (void *)out, (void *)index, (void *)weight);
            return;
        }
        llaisys::ops::embedding(out->tensor, index->tensor, weight->tensor);
    }

    void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias) {
        if (!out || !in || !weight) {
            std::fprintf(stderr, "[llaisys] llaisysLinear: nullptr arg(s): out=%p in=%p weight=%p bias=%p\n",
                         (void *)out, (void *)in, (void *)weight, (void *)bias);
            return;
        }
        // bias is optional
        llaisys::tensor_t bias_t = bias ? bias->tensor : llaisys::tensor_t{};
        llaisys::ops::linear(out->tensor, in->tensor, weight->tensor, bias_t);
    }

    void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in) {
        if (!out || !in) {
            std::fprintf(stderr, "[llaisys] llaisysRearrange: nullptr arg(s): out=%p in=%p\n", (void *)out, (void *)in);
            return;
        }
        llaisys::ops::rearrange(out->tensor, in->tensor);
    }

    void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps) {
        if (!out || !in || !weight) {
            std::fprintf(stderr, "[llaisys] llaisysRmsNorm: nullptr arg(s): out=%p in=%p weight=%p\n",
                         (void *)out, (void *)in, (void *)weight);
            return;
        }
        llaisys::ops::rms_norm(out->tensor, in->tensor, weight->tensor, eps);
    }

    void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta) {
        if (!out || !in || !pos_ids) {
            std::fprintf(stderr, "[llaisys] llaisysROPE: nullptr arg(s): out=%p in=%p pos_ids=%p\n",
                         (void *)out, (void *)in, (void *)pos_ids);
            return;
        }
        llaisys::ops::rope(out->tensor, in->tensor, pos_ids->tensor, theta);
    }

    void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale) {
        if (!attn_val || !q || !k || !v) {
            std::fprintf(stderr, "[llaisys] llaisysSelfAttention: nullptr arg(s): attn_val=%p q=%p k=%p v=%p\n",
                         (void *)attn_val, (void *)q, (void *)k, (void *)v);
            return;
        }
        llaisys::ops::self_attention(attn_val->tensor, q->tensor, k->tensor, v->tensor, scale);
    }

    void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up) {
        if (!out || !gate || !up) {
            std::fprintf(stderr, "[llaisys] llaisysSwiGLU: nullptr arg(s): out=%p gate=%p up=%p\n",
                         (void *)out, (void *)gate, (void *)up);
            return;
        }
        llaisys::ops::swiglu(out->tensor, gate->tensor, up->tensor);
    }

} // __C