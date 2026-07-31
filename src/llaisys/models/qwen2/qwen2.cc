#include "llaisys/models/qwen2.h"
#include "llaisys/ops.h"
#include "llaisys/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <new>
#include <unordered_set>
#include <vector>

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    llaisysDeviceType_t device{};
    std::vector<int> device_ids;

    LlaisysQwen2Weights w{};
    bool weights_inited{false};

    // KV cache (per-layer)
    llaisysTensor_t *k_cache{nullptr}; // [maxseq, nkvh, dh]
    llaisysTensor_t *v_cache{nullptr}; // [maxseq, nkvh, dh]
    size_t cached_len{0};
    std::vector<int64_t> cached_tokens; // 用于判断前缀是否一致
};

static int pick_device_id(const LlaisysQwen2Model *m) {
    if (!m) {
        return 0;
    }
    if (!m->device_ids.empty()) {
        return m->device_ids[0];
    }
    return 0;
}

static bool meta_sane(const LlaisysQwen2Meta *m) {
    if (!m) {
        return false;
    }
    if (m->nlayer == 0) {
        return false;
    }
    if (m->hs == 0 || m->nh == 0 || m->dh == 0) {
        return false;
    }
    if (m->hs != m->nh * m->dh) {
        return false;
    }
    if (m->nkvh == 0 || (m->nh % m->nkvh) != 0) {
        return false;
    }
    if (m->di == 0 || m->voc == 0) {
        return false;
    }
    if (m->maxseq == 0) {
        return false;
    }
    return true;
}

static void weights_alloc(LlaisysQwen2Model *model) {
    const size_t L = model->meta.nlayer;

    model->w.attn_norm_w = new llaisysTensor_t[L]();
    model->w.attn_q_w = new llaisysTensor_t[L]();
    model->w.attn_q_b = new llaisysTensor_t[L]();
    model->w.attn_k_w = new llaisysTensor_t[L]();
    model->w.attn_k_b = new llaisysTensor_t[L]();
    model->w.attn_v_w = new llaisysTensor_t[L]();
    model->w.attn_v_b = new llaisysTensor_t[L]();
    model->w.attn_o_w = new llaisysTensor_t[L]();

    model->w.mlp_norm_w = new llaisysTensor_t[L]();
    model->w.mlp_gate_w = new llaisysTensor_t[L]();
    model->w.mlp_up_w = new llaisysTensor_t[L]();
    model->w.mlp_down_w = new llaisysTensor_t[L]();

    model->weights_inited = true;
}

static void kv_alloc(LlaisysQwen2Model *model) {
    const size_t L = model->meta.nlayer;
    model->k_cache = new llaisysTensor_t[L]();
    model->v_cache = new llaisysTensor_t[L]();

    const int devid = pick_device_id(model);
    for (size_t i = 0; i < L; ++i) {
        size_t shape[3]{model->meta.maxseq, model->meta.nkvh, model->meta.dh};
        model->k_cache[i] = tensorCreate(shape, 3, model->meta.dtype, model->device, devid);
        model->v_cache[i] = tensorCreate(shape, 3, model->meta.dtype, model->device, devid);
    }
    model->cached_len = 0;
    model->cached_tokens.clear();
}

static void kv_free(LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }

    if (model->k_cache) {
        for (size_t i = 0; i < model->meta.nlayer; ++i) {
            if (model->k_cache[i]) {
                tensorDestroy(model->k_cache[i]);
            }
        }
        delete[] model->k_cache;
        model->k_cache = nullptr;
    }
    if (model->v_cache) {
        for (size_t i = 0; i < model->meta.nlayer; ++i) {
            if (model->v_cache[i]) {
                tensorDestroy(model->v_cache[i]);
            }
        }
        delete[] model->v_cache;
        model->v_cache = nullptr;
    }

    model->cached_len = 0;
    model->cached_tokens.clear();
}

static void destroy_unique_tensor(std::unordered_set<void *> &seen, llaisysTensor_t t) {
    if (!t) {
        return;
    }
    void *p = (void *)t;
    if (seen.insert(p).second) {
        tensorDestroy(t);
    }
}

static void weights_free_and_destroy_tensors(LlaisysQwen2Model *model) {
    if (!model || !model->weights_inited) {
        return;
    }

    std::unordered_set<void *> seen;
    destroy_unique_tensor(seen, model->w.in_embed);
    destroy_unique_tensor(seen, model->w.out_embed);
    destroy_unique_tensor(seen, model->w.out_norm_w);

    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        destroy_unique_tensor(seen, model->w.attn_norm_w[i]);
        destroy_unique_tensor(seen, model->w.attn_q_w[i]);
        destroy_unique_tensor(seen, model->w.attn_q_b[i]);
        destroy_unique_tensor(seen, model->w.attn_k_w[i]);
        destroy_unique_tensor(seen, model->w.attn_k_b[i]);
        destroy_unique_tensor(seen, model->w.attn_v_w[i]);
        destroy_unique_tensor(seen, model->w.attn_v_b[i]);
        destroy_unique_tensor(seen, model->w.attn_o_w[i]);

        destroy_unique_tensor(seen, model->w.mlp_norm_w[i]);
        destroy_unique_tensor(seen, model->w.mlp_gate_w[i]);
        destroy_unique_tensor(seen, model->w.mlp_up_w[i]);
        destroy_unique_tensor(seen, model->w.mlp_down_w[i]);
    }

    delete[] model->w.attn_norm_w;
    delete[] model->w.attn_q_w;
    delete[] model->w.attn_q_b;
    delete[] model->w.attn_k_w;
    delete[] model->w.attn_k_b;
    delete[] model->w.attn_v_w;
    delete[] model->w.attn_v_b;
    delete[] model->w.attn_o_w;
    delete[] model->w.mlp_norm_w;
    delete[] model->w.mlp_gate_w;
    delete[] model->w.mlp_up_w;
    delete[] model->w.mlp_down_w;

    model->w = LlaisysQwen2Weights{};
    model->weights_inited = false;
}

static bool weights_ready(const LlaisysQwen2Model *m) {
    if (!m) {
        return false;
    }
    if (!m->w.in_embed || !m->w.out_embed || !m->w.out_norm_w) {
        return false;
    }

    for (size_t i = 0; i < m->meta.nlayer; ++i) {
        if (!m->w.attn_norm_w[i]) {
            return false;
        }
        if (!m->w.attn_q_w[i] || !m->w.attn_k_w[i] || !m->w.attn_v_w[i] || !m->w.attn_o_w[i]) {
            return false;
        }
        if (!m->w.mlp_norm_w[i] || !m->w.mlp_gate_w[i] || !m->w.mlp_up_w[i] || !m->w.mlp_down_w[i]) {
            return false;
        }
    }
    return true;
}

struct TensorGuard {
    llaisysTensor_t t{nullptr};
    TensorGuard() = default;
    explicit TensorGuard(llaisysTensor_t x) : t(x) {}
    TensorGuard(const TensorGuard &) = delete;
    TensorGuard &operator=(const TensorGuard &) = delete;
    TensorGuard(TensorGuard &&o) noexcept : t(o.t) { o.t = nullptr; }
    TensorGuard &operator=(TensorGuard &&o) noexcept {
        if (this != &o) {
            if (t) {
                tensorDestroy(t);
            }
            t = o.t;
            o.t = nullptr;
        }
        return *this;
    }
    ~TensorGuard() {
        if (t) {
            tensorDestroy(t);
        }
    }
    operator llaisysTensor_t() const { return t; }
};

static llaisysTensor_t make_tensor_1d(size_t n, llaisysDataType_t dt, llaisysDeviceType_t dev, int devid) {
    size_t shape[1]{n};
    return tensorCreate(shape, 1, dt, dev, devid);
}
static llaisysTensor_t make_tensor_2d(size_t a, size_t b, llaisysDataType_t dt, llaisysDeviceType_t dev, int devid) {
    size_t shape[2]{a, b};
    return tensorCreate(shape, 2, dt, dev, devid);
}
static llaisysTensor_t make_tensor_3d(size_t a, size_t b, size_t c, llaisysDataType_t dt, llaisysDeviceType_t dev, int devid) {
    size_t shape[3]{a, b, c};
    return tensorCreate(shape, 3, dt, dev, devid);
}

static void reset_cache(LlaisysQwen2Model *m) {
    m->cached_len = 0;
    m->cached_tokens.clear();
}

extern "C" {

__export LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {

    if (!meta_sane(meta)) {
        return nullptr;
    }

    auto *m = new (std::nothrow) LlaisysQwen2Model();
    if (!m) {
        return nullptr;
    }

    m->meta = *meta;
    m->device = device;
    if (device_ids && ndevice > 0) {
        m->device_ids.assign(device_ids, device_ids + ndevice);
    }

    weights_alloc(m);
    kv_alloc(m);
    return m;
}

__export void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }
    kv_free(model);
    weights_free_and_destroy_tensors(model);
    delete model;
}

__export LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
    if (!model) {
        return nullptr;
    }
    return &model->w;
}

__export int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    // ---- switches ----
    static const bool DEBUG_LAYERS = false;   // 逐层日志（默认关）
    static const bool STAGE_MARKS = false;     // 阶段打点（默认开，用于定位卡住）
    static const bool ENABLE_KV_WRITE = true; // KV cache 写入（默认开；卡住就先改 false 验证）
    static const bool FORCE_DV_EQ_D = false;  // 仅用于定位（默认关）

    auto MARK = [&](const char *msg) {
        if (!STAGE_MARKS) {
            return;
        }
        std::fprintf(stderr, "[infer] %s\n", msg);
        std::fflush(stderr);
    };
    auto MARK_L = [&](size_t l, const char *msg) {
        if (!DEBUG_LAYERS) {
            return;
        }
        std::fprintf(stderr, "[infer][L=%zu] %s\n", l, msg);
        std::fflush(stderr);
    };

    auto dump_shape = [&](const char *name, llaisysTensor_t t) {
        if (!t) {
            std::fprintf(stderr, "[shape] %s = <null>\n", name);
            return;
        }
        size_t nd = tensorGetNdim(t);
        std::vector<size_t> sh(nd);
        std::vector<ptrdiff_t> st(nd);
        tensorGetShape(t, sh.data());
        tensorGetStrides(t, st.data());
        std::fprintf(stderr, "[shape] %s ndim=%zu shape=[", name, nd);
        for (size_t i = 0; i < nd; ++i) {
            std::fprintf(stderr, "%zu%s", sh[i], (i + 1 == nd) ? "" : ",");
        }
        std::fprintf(stderr, "] strides=[");
        for (size_t i = 0; i < nd; ++i) {
            std::fprintf(stderr, "%td%s", st[i], (i + 1 == nd) ? "" : ",");
        }
        std::fprintf(stderr, "] contiguous=%u dtype=%d dev=%d devid=%d\n",
                     (unsigned)tensorIsContiguous(t),
                     (int)tensorGetDataType(t),
                     (int)tensorGetDeviceType(t),
                     (int)tensorGetDeviceId(t));
        std::fflush(stderr);
    };

    try {
        MARK("enter");

        if (!model || !token_ids || ntoken == 0) {
            std::fprintf(stderr, "[infer] bad args: model=%p token_ids=%p ntoken=%zu\n",
                         (void *)model, (void *)token_ids, ntoken);
            return -1;
        }
        if (!weights_ready(model)) {
            MARK("weights not ready");
            return -2;
        }
        if (ntoken > model->meta.maxseq) {
            std::fprintf(stderr, "[infer] ntoken too long: ntoken=%zu maxseq=%zu\n", ntoken, model->meta.maxseq);
            return -3;
        }

        const auto &meta = model->meta;
        const int devid = pick_device_id(model);

        std::fprintf(stderr,
                     "[infer] device=%d devid=%d ntoken=%zu cached_len=%zu dtype=%d nlayer=%zu hs=%zu nh=%zu nkvh=%zu dh=%zu di=%zu voc=%zu maxseq=%zu\n",
                     (int)model->device, devid, ntoken, model->cached_len, (int)meta.dtype, meta.nlayer, meta.hs, meta.nh,
                     meta.nkvh, meta.dh, meta.di, meta.voc, meta.maxseq);
        std::fflush(stderr);

        // ---------- decide incremental ----------
        size_t start = 0;
        if (model->cached_len > 0 && ntoken >= model->cached_len) {
            bool prefix_ok = true;
            for (size_t i = 0; i < model->cached_len; ++i) {
                if (i >= model->cached_tokens.size() || model->cached_tokens[i] != token_ids[i]) {
                    prefix_ok = false;
                    break;
                }
            }
            if (prefix_ok) {
                start = model->cached_len;
            }
        }
        if (start == 0 && model->cached_len != 0) {
            MARK("prefix mismatch -> reset_cache");
            reset_cache(model);
        }

        const size_t new_len = ntoken - start;
        std::fprintf(stderr, "[infer] start=%zu new_len=%zu\n", start, new_len);
        std::fflush(stderr);
        if (new_len == 0) {
            return -4;
        }

        const float attn_scale = 1.0f / std::sqrt((float)meta.dh);

        // token ids new: [new_len] int64
        MARK("alloc/load tokens");
        TensorGuard t_tok(make_tensor_1d(new_len, LLAISYS_DTYPE_I64, model->device, devid));
        tensorLoad(t_tok.t, token_ids + start);

        // pos ids absolute: [new_len] int64
        MARK("alloc/load pos");
        std::vector<int64_t> pos(new_len);
        for (size_t i = 0; i < new_len; ++i) {
            pos[i] = (int64_t)(start + i);
        }
        TensorGuard t_pos(make_tensor_1d(new_len, LLAISYS_DTYPE_I64, model->device, devid));
        tensorLoad(t_pos.t, pos.data());

        // x: [new_len, hs]
        MARK("embedding");
        TensorGuard x(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
        llaisysEmbedding(x.t, t_tok.t, model->w.in_embed);

        MARK("before blocks");

        // blocks
        for (size_t l = 0; l < meta.nlayer; ++l) {
            MARK_L(l, "begin");

            if (l == 0) {
                MARK("L0: attn rmsnorm");
            }
            TensorGuard xn(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
            llaisysRmsNorm(xn.t, x.t, model->w.attn_norm_w[l], meta.epsilon);

            if (l == 0) {
                MARK("L0: qkv linear");
            }
            TensorGuard q2(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
            TensorGuard k2(make_tensor_2d(new_len, meta.nkvh * meta.dh, meta.dtype, model->device, devid));
            TensorGuard v2(make_tensor_2d(new_len, meta.nkvh * meta.dh, meta.dtype, model->device, devid));
            llaisysLinear(q2.t, xn.t, model->w.attn_q_w[l], model->w.attn_q_b ? model->w.attn_q_b[l] : nullptr);
            llaisysLinear(k2.t, xn.t, model->w.attn_k_w[l], model->w.attn_k_b ? model->w.attn_k_b[l] : nullptr);
            llaisysLinear(v2.t, xn.t, model->w.attn_v_w[l], model->w.attn_v_b ? model->w.attn_v_b[l] : nullptr);

            MARK_L(l, "view qkv -> 3d");
            size_t qshape[3]{new_len, meta.nh, meta.dh};
            size_t kvshape[3]{new_len, meta.nkvh, meta.dh};
            TensorGuard q3(tensorView(q2.t, qshape, 3));
            TensorGuard k3(tensorView(k2.t, kvshape, 3));
            TensorGuard v3(tensorView(v2.t, kvshape, 3));

            if (l == 0) {
                MARK("L0: rope");
            }
            TensorGuard q_rope(make_tensor_3d(new_len, meta.nh, meta.dh, meta.dtype, model->device, devid));
            TensorGuard k_rope(make_tensor_3d(new_len, meta.nkvh, meta.dh, meta.dtype, model->device, devid));
            llaisysROPE(q_rope.t, q3.t, t_pos.t, meta.theta);
            llaisysROPE(k_rope.t, k3.t, t_pos.t, meta.theta);

            if (ENABLE_KV_WRITE) {
                if (l == 0) {
                    MARK("L0: write kv cache");
                }
                TensorGuard k_dst(tensorSlice(model->k_cache[l], 0, start, start + new_len));
                TensorGuard v_dst(tensorSlice(model->v_cache[l], 0, start, start + new_len));
                llaisysRearrange(k_dst.t, k_rope.t);
                llaisysRearrange(v_dst.t, v3.t);
            }

            if (l == 0) {
                MARK("L0: k/v total slice");
            }
            TensorGuard k_total(tensorSlice(model->k_cache[l], 0, 0, start + new_len));
            TensorGuard v_total(tensorSlice(model->v_cache[l], 0, 0, start + new_len));

            if (l == 0) {
                MARK("L0: self_attention");
            }
            // dv 在你的 meta 里通常等于 dh；如果不是，且你的 self_attention 实现很慢，
            // 可临时开 FORCE_DV_EQ_D 做定位。
            const size_t dv = FORCE_DV_EQ_D ? meta.dh : meta.dh;

            TensorGuard attn_val(make_tensor_3d(new_len, meta.nh, dv, meta.dtype, model->device, devid));
            if (l == 0) {
                dump_shape("q_rope", q_rope.t);
                dump_shape("k_total", k_total.t);
                dump_shape("v_total", v_total.t);
                dump_shape("attn_val(out, pre)", attn_val.t);
            }

            llaisysSelfAttention(attn_val.t, q_rope.t, k_total.t, v_total.t, attn_scale);

            if (l == 0) {
                dump_shape("attn_val(out, post)", attn_val.t);
            }

            MARK_L(l, "view attn -> 2d");
            size_t x2shape[2]{new_len, meta.hs};
            TensorGuard attn_2d(tensorView(attn_val.t, x2shape, 2));

            if (l == 0) {
                MARK("L0: attn out linear");
            }
            TensorGuard attn_out(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
            llaisysLinear(attn_out.t, attn_2d.t, model->w.attn_o_w[l], nullptr);

            MARK_L(l, "residual add 1");
            TensorGuard x1(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
            llaisysAdd(x1.t, x.t, attn_out.t);

            if (l == 0) {
                MARK("L0: mlp");
            }
            TensorGuard x1n(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
            llaisysRmsNorm(x1n.t, x1.t, model->w.mlp_norm_w[l], meta.epsilon);

            TensorGuard gate(make_tensor_2d(new_len, meta.di, meta.dtype, model->device, devid));
            TensorGuard up(make_tensor_2d(new_len, meta.di, meta.dtype, model->device, devid));
            llaisysLinear(gate.t, x1n.t, model->w.mlp_gate_w[l], nullptr);
            llaisysLinear(up.t, x1n.t, model->w.mlp_up_w[l], nullptr);

            TensorGuard act(make_tensor_2d(new_len, meta.di, meta.dtype, model->device, devid));
            llaisysSwiGLU(act.t, gate.t, up.t);

            TensorGuard mlp_out(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
            llaisysLinear(mlp_out.t, act.t, model->w.mlp_down_w[l], nullptr);

            TensorGuard x2(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
            llaisysAdd(x2.t, x1.t, mlp_out.t);

            x = std::move(x2);
            MARK_L(l, "end");
        }

        MARK("after blocks");

        MARK("final norm");
        TensorGuard xnorm(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
        llaisysRmsNorm(xnorm.t, x.t, model->w.out_norm_w, meta.epsilon);

        MARK("logits linear");
        TensorGuard logits(make_tensor_2d(new_len, meta.voc, meta.dtype, model->device, devid));
        llaisysLinear(logits.t, xnorm.t, model->w.out_embed, nullptr);

        MARK("last row slice + view");
        TensorGuard last2d(tensorSlice(logits.t, 0, new_len - 1, new_len));
        size_t last_shape[1]{meta.voc};
        TensorGuard last1d(tensorView(last2d.t, last_shape, 1));

        MARK("argmax");
        TensorGuard max_idx(make_tensor_1d(1, LLAISYS_DTYPE_I64, model->device, devid));
        TensorGuard max_val(make_tensor_1d(1, meta.dtype, model->device, devid));
        llaisysArgmax(max_idx.t, max_val.t, last1d.t);

        MARK("read max_idx");
        void *p = tensorGetData(max_idx.t);
        if (!p) {
            return -100;
        }
        int64_t out = *reinterpret_cast<int64_t *>(p);

        MARK("update cache");
        model->cached_len = ntoken;
        model->cached_tokens.assign(token_ids, token_ids + ntoken);

        std::fprintf(stderr, "[infer] return token=%lld\n", (long long)out);
        std::fflush(stderr);
        return out;

    } catch (const std::exception &e) {
        std::fprintf(stderr, "[infer] exception caught: %s\n", e.what());
        std::fflush(stderr);
        return -999;
    } catch (...) {
        std::fprintf(stderr, "[infer] unknown exception caught\n");
        std::fflush(stderr);
        return -998;
    }
}

} // extern "C"