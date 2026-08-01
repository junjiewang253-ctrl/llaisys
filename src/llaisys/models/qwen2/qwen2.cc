#include "llaisys/models/qwen2.h"
#include "llaisys/error.h"
#include "llaisys/ops.h"
#include "llaisys/runtime.h"
#include "llaisys/tensor.h"

#include "../../../utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#undef __C
#include <immintrin.h>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef LLAISYS_USE_DNNL
#include <oneapi/dnnl/dnnl.hpp>
#endif

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
    LlaisysQwen2Trace trace{};
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

static const LlaisysRuntimeAPI *model_runtime(const LlaisysQwen2Model *model) {
    if (!model || model->device == LLAISYS_DEVICE_CPU) {
        return nullptr;
    }
    llaisysSetContextRuntime(model->device, pick_device_id(model));
    const auto *api = llaisysGetRuntimeAPI(model->device);
    if (!api || llaisysGetLastErrorCode() != LLAISYS_STATUS_SUCCESS) {
        throw std::runtime_error("Qwen2 device runtime is unavailable");
    }
    api->set_device(pick_device_id(model));
    return api;
}

static void copy_tensor_bytes(LlaisysQwen2Model *model,
                              void *destination,
                              const void *source,
                              size_t bytes,
                              llaisysMemcpyKind_t kind) {
    if (bytes == 0) {
        return;
    }
    if (!destination || !source) {
        throw std::runtime_error("Qwen2 tensor copy received a null pointer");
    }
    if (model->device == LLAISYS_DEVICE_CPU) {
        std::memcpy(destination, source, bytes);
        return;
    }
    const auto *api = model_runtime(model);
    api->device_synchronize();
    if (llaisysGetLastErrorCode() != LLAISYS_STATUS_SUCCESS) {
        throw std::runtime_error("Qwen2 device synchronization before copy failed");
    }
    api->memcpy_sync(destination, source, bytes, kind);
    if (llaisysGetLastErrorCode() != LLAISYS_STATUS_SUCCESS) {
        throw std::runtime_error("Qwen2 device copy failed");
    }
    api->device_synchronize();
    if (llaisysGetLastErrorCode() != LLAISYS_STATUS_SUCCESS) {
        throw std::runtime_error("Qwen2 device synchronization after copy failed");
    }
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
    if (m->dtype != LLAISYS_DTYPE_F32 &&
        m->dtype != LLAISYS_DTYPE_F16 &&
        m->dtype != LLAISYS_DTYPE_BF16) {
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

static size_t checked_multiply(size_t lhs, size_t rhs, const char *message) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::overflow_error(message);
    }
    return lhs * rhs;
}

static size_t kv_payload_bytes(const LlaisysQwen2Model *model) {
    if (!model) {
        return 0;
    }
    size_t elements = checked_multiply(
        2, model->meta.nlayer, "Qwen2 KV layer count overflow");
    elements = checked_multiply(
        elements, model->meta.maxseq, "Qwen2 KV capacity overflow");
    elements = checked_multiply(
        elements, model->meta.nkvh, "Qwen2 KV head count overflow");
    elements = checked_multiply(
        elements, model->meta.dh, "Qwen2 KV head dimension overflow");
    return checked_multiply(
        elements, llaisys::utils::dsize(model->meta.dtype),
        "Qwen2 KV byte size overflow");
}

static void kv_alloc(LlaisysQwen2Model *model) {
    if (!model) {
        throw std::invalid_argument("Qwen2 KV model is null");
    }
    // Validate the complete byte expression before making any allocation.
    (void)kv_payload_bytes(model);
    model->k_cache = new llaisysTensor_t[model->meta.nlayer]();
    model->v_cache = new llaisysTensor_t[model->meta.nlayer]();
    const int device_id = pick_device_id(model);
    for (size_t layer = 0; layer < model->meta.nlayer; ++layer) {
        model->k_cache[layer] = make_tensor_3d(
            model->meta.maxseq, model->meta.nkvh, model->meta.dh,
            model->meta.dtype, model->device, device_id);
        model->v_cache[layer] = make_tensor_3d(
            model->meta.maxseq, model->meta.nkvh, model->meta.dh,
            model->meta.dtype, model->device, device_id);
        if (!model->k_cache[layer] || !model->v_cache[layer]) {
            throw std::runtime_error("failed to allocate Qwen2 KV cache");
        }
    }
}

static void trace_clear(LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }
    tensorDestroy(model->trace.embedding);
    tensorDestroy(model->trace.final_norm);
    tensorDestroy(model->trace.logits);
    tensorDestroy(model->trace.diagnostic_attn_norm);
    tensorDestroy(model->trace.diagnostic_q);
    tensorDestroy(model->trace.diagnostic_k);
    tensorDestroy(model->trace.diagnostic_v);
    tensorDestroy(model->trace.diagnostic_q_rope);
    tensorDestroy(model->trace.diagnostic_k_rope);
    tensorDestroy(model->trace.diagnostic_attention_scores);
    tensorDestroy(model->trace.diagnostic_attention_probabilities);
    tensorDestroy(model->trace.diagnostic_attn_value);
    tensorDestroy(model->trace.diagnostic_post_attention);
    tensorDestroy(model->trace.diagnostic_mlp_norm);
    tensorDestroy(model->trace.diagnostic_gate);
    tensorDestroy(model->trace.diagnostic_up);
    tensorDestroy(model->trace.diagnostic_activation);
    tensorDestroy(model->trace.diagnostic_mlp_out);
    if (model->trace.attention_out) {
        for (size_t i = 0; i < model->meta.nlayer; ++i) {
            tensorDestroy(model->trace.attention_out[i]);
        }
        delete[] model->trace.attention_out;
    }
    if (model->trace.layer_output) {
        for (size_t i = 0; i < model->meta.nlayer; ++i) {
            tensorDestroy(model->trace.layer_output[i]);
        }
        delete[] model->trace.layer_output;
    }
    model->trace = LlaisysQwen2Trace{};
}

static llaisysTensor_t trace_copy(llaisysTensor_t tensor) {
    auto result = tensorContiguous(tensor);
    if (!result || llaisysGetLastErrorCode() != LLAISYS_STATUS_SUCCESS) {
        throw std::runtime_error("failed to capture Qwen2 trace tensor");
    }
    return result;
}

static void require_op_success(const char *stage) {
    if (llaisysGetLastErrorCode() != LLAISYS_STATUS_SUCCESS) {
        const char *message = llaisysGetLastErrorMessage();
        throw std::runtime_error(
            std::string(stage) + ": " + (message ? message : "unknown error"));
    }
}

static void qwen2_rope_bf16(llaisysTensor_t out,
                            llaisysTensor_t input,
                            llaisysTensor_t positions,
                            float theta,
                            size_t sequence,
                            size_t heads,
                            size_t dimension) {
    auto *destination =
        reinterpret_cast<llaisys::bf16_t *>(tensorGetData(out));
    const auto *source =
        reinterpret_cast<const llaisys::bf16_t *>(tensorGetData(input));
    const auto *position =
        reinterpret_cast<const int64_t *>(tensorGetData(positions));
    const size_t half = dimension / 2;
    for (size_t s = 0; s < sequence; ++s) {
        for (size_t d = 0; d < half; ++d) {
            const float frequency =
                static_cast<float>(position[s]) /
                std::pow(theta, 2.0f * static_cast<float>(d) /
                                    static_cast<float>(dimension));
            const auto cosine =
                llaisys::utils::cast<llaisys::bf16_t>(std::cos(frequency));
            const auto sine =
                llaisys::utils::cast<llaisys::bf16_t>(std::sin(frequency));
            const float c = llaisys::utils::cast<float>(cosine);
            const float sn = llaisys::utils::cast<float>(sine);
            for (size_t h = 0; h < heads; ++h) {
                const size_t left = (s * heads + h) * dimension + d;
                const size_t right = left + half;
                const float x1 = llaisys::utils::cast<float>(source[left]);
                const float x2 = llaisys::utils::cast<float>(source[right]);
                const auto left_cos =
                    llaisys::utils::cast<llaisys::bf16_t>(x1 * c);
                const auto right_sin =
                    llaisys::utils::cast<llaisys::bf16_t>(x2 * sn);
                const auto right_cos =
                    llaisys::utils::cast<llaisys::bf16_t>(x2 * c);
                const auto left_sin =
                    llaisys::utils::cast<llaisys::bf16_t>(x1 * sn);
                destination[left] = llaisys::utils::cast<llaisys::bf16_t>(
                    llaisys::utils::cast<float>(left_cos) -
                    llaisys::utils::cast<float>(right_sin));
                destination[right] = llaisys::utils::cast<llaisys::bf16_t>(
                    llaisys::utils::cast<float>(right_cos) +
                    llaisys::utils::cast<float>(left_sin));
            }
        }
    }
}

static void qwen2_swiglu_bf16(llaisysTensor_t out,
                              llaisysTensor_t gate,
                              llaisysTensor_t up,
                              size_t elements) {
    auto *destination =
        reinterpret_cast<llaisys::bf16_t *>(tensorGetData(out));
    const auto *gate_data =
        reinterpret_cast<const llaisys::bf16_t *>(tensorGetData(gate));
    const auto *up_data =
        reinterpret_cast<const llaisys::bf16_t *>(tensorGetData(up));
    for (size_t index = 0; index < elements; ++index) {
        const float gate_value =
            llaisys::utils::cast<float>(gate_data[index]);
        const float up_value =
            llaisys::utils::cast<float>(up_data[index]);
        const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
        const auto silu = llaisys::utils::cast<llaisys::bf16_t>(
            gate_value * sigmoid);
        destination[index] = llaisys::utils::cast<llaisys::bf16_t>(
            llaisys::utils::cast<float>(silu) * up_value);
    }
}

#ifdef LLAISYS_USE_DNNL
static void qwen2_attention_bf16(llaisysTensor_t out,
                                 llaisysTensor_t query,
                                 llaisysTensor_t key,
                                 llaisysTensor_t value,
                                 size_t query_length,
                                 size_t total_length,
                                 size_t heads,
                                 size_t kv_heads,
                                 size_t dimension,
                                 size_t value_dimension,
                                 float scale,
                                 llaisysTensor_t *score_trace,
                                 llaisysTensor_t *probability_trace) {
    const auto *query_data =
        reinterpret_cast<const llaisys::bf16_t *>(tensorGetData(query));
    const auto *key_data =
        reinterpret_cast<const llaisys::bf16_t *>(tensorGetData(key));
    const auto *value_data =
        reinterpret_cast<const llaisys::bf16_t *>(tensorGetData(value));
    auto *destination =
        reinterpret_cast<llaisys::bf16_t *>(tensorGetData(out));
    const size_t group = heads / kv_heads;
    std::vector<llaisys::bf16_t> queries(
        heads * query_length * dimension);
    std::vector<llaisys::bf16_t> keys(
        heads * total_length * dimension);
    std::vector<llaisys::bf16_t> values(
        heads * total_length * value_dimension);
    for (size_t head = 0; head < heads; ++head) {
        const size_t kv_head = head / group;
        for (size_t token = 0; token < query_length; ++token) {
            std::copy_n(
                query_data + (token * heads + head) * dimension,
                dimension,
                queries.data() +
                    (head * query_length + token) * dimension);
        }
        for (size_t token = 0; token < total_length; ++token) {
            std::copy_n(
                key_data + (token * kv_heads + kv_head) * dimension,
                dimension,
                keys.data() +
                    (head * total_length + token) * dimension);
            std::copy_n(
                value_data +
                    (token * kv_heads + kv_head) * value_dimension,
                value_dimension,
                values.data() +
                    (head * total_length + token) * value_dimension);
        }
    }

    static const dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    static dnnl::stream stream(engine);
    const dnnl::memory::dims query_dims{
        static_cast<dnnl_dim_t>(heads),
        static_cast<dnnl_dim_t>(query_length),
        static_cast<dnnl_dim_t>(dimension)};
    const dnnl::memory::dims query_strides{
        static_cast<dnnl_dim_t>(query_length * dimension),
        static_cast<dnnl_dim_t>(dimension), 1};
    const dnnl::memory::dims key_dims{
        static_cast<dnnl_dim_t>(heads),
        static_cast<dnnl_dim_t>(dimension),
        static_cast<dnnl_dim_t>(total_length)};
    const dnnl::memory::dims key_strides{
        static_cast<dnnl_dim_t>(total_length * dimension), 1,
        static_cast<dnnl_dim_t>(dimension)};
    const dnnl::memory::dims score_dims{
        static_cast<dnnl_dim_t>(heads),
        static_cast<dnnl_dim_t>(query_length),
        static_cast<dnnl_dim_t>(total_length)};
    const dnnl::memory::dims score_strides{
        static_cast<dnnl_dim_t>(query_length * total_length),
        static_cast<dnnl_dim_t>(total_length), 1};
    const auto query_desc = dnnl::memory::desc(
        query_dims, dnnl::memory::data_type::bf16, query_strides);
    const auto key_desc = dnnl::memory::desc(
        key_dims, dnnl::memory::data_type::bf16, key_strides);
    const auto score_desc = dnnl::memory::desc(
        score_dims, dnnl::memory::data_type::bf16, score_strides);
    std::vector<llaisys::bf16_t> scores(
        heads * query_length * total_length);
    const dnnl::matmul::primitive_desc score_primitive_desc(
        engine, query_desc, key_desc, score_desc);
    dnnl::matmul(score_primitive_desc)
        .execute(
            stream,
            {{DNNL_ARG_SRC,
                 dnnl::memory(query_desc, engine, queries.data())},
             {DNNL_ARG_WEIGHTS,
                 dnnl::memory(key_desc, engine, keys.data())},
             {DNNL_ARG_DST,
                 dnnl::memory(score_desc, engine, scores.data())}});
    stream.wait();

    std::vector<float> scaled_scores(
        heads * query_length * total_length);
    const size_t prefix = total_length - query_length;
    for (size_t head = 0; head < heads; ++head) {
        for (size_t row = 0; row < query_length; ++row) {
            const size_t offset =
                (head * query_length + row) * total_length;
            const size_t last_visible = prefix + row;
            for (size_t column = 0; column <= last_visible; ++column) {
                const auto scaled = llaisys::utils::cast<llaisys::bf16_t>(
                    llaisys::utils::cast<float>(scores[offset + column]) *
                    scale);
                scores[offset + column] = scaled;
                scaled_scores[offset + column] =
                    llaisys::utils::cast<float>(scaled);
            }
            for (size_t column = last_visible + 1;
                 column < total_length; ++column) {
                scores[offset + column] =
                    llaisys::utils::cast<llaisys::bf16_t>(
                        -std::numeric_limits<float>::infinity());
                scaled_scores[offset + column] =
                    -std::numeric_limits<float>::infinity();
            }
        }
    }
    const auto probability_f32_desc = dnnl::memory::desc(
        score_dims, dnnl::memory::data_type::f32, score_strides);
    std::vector<float> probabilities_f32(
        heads * query_length * total_length);
    const auto softmax_primitive_desc =
        dnnl::softmax_forward::primitive_desc(
            engine, dnnl::prop_kind::forward_inference,
            dnnl::algorithm::softmax_accurate,
            probability_f32_desc, probability_f32_desc, 2);
    dnnl::softmax_forward(softmax_primitive_desc)
        .execute(
            stream,
            {{DNNL_ARG_SRC,
                 dnnl::memory(
                     probability_f32_desc, engine,
                     scaled_scores.data())},
             {DNNL_ARG_DST,
                 dnnl::memory(
                     probability_f32_desc, engine,
                     probabilities_f32.data())}});
    stream.wait();
    std::vector<llaisys::bf16_t> probabilities(
        heads * query_length * total_length);
    for (size_t index = 0; index < probabilities.size(); ++index) {
        probabilities[index] =
            llaisys::utils::cast<llaisys::bf16_t>(
                probabilities_f32[index]);
    }
    if (score_trace) {
        *score_trace = make_tensor_3d(
            heads, query_length, total_length, LLAISYS_DTYPE_BF16,
            LLAISYS_DEVICE_CPU, 0);
        tensorLoad(*score_trace, scores.data());
        require_op_success("attention score trace");
    }
    if (probability_trace) {
        *probability_trace = make_tensor_3d(
            heads, query_length, total_length, LLAISYS_DTYPE_BF16,
            LLAISYS_DEVICE_CPU, 0);
        tensorLoad(*probability_trace, probabilities.data());
        require_op_success("attention probability trace");
    }

    const dnnl::memory::dims value_source_dims{
        static_cast<dnnl_dim_t>(heads),
        static_cast<dnnl_dim_t>(total_length),
        static_cast<dnnl_dim_t>(value_dimension)};
    const dnnl::memory::dims value_source_strides{
        static_cast<dnnl_dim_t>(total_length * value_dimension),
        static_cast<dnnl_dim_t>(value_dimension), 1};
    const dnnl::memory::dims attended_dims{
        static_cast<dnnl_dim_t>(heads),
        static_cast<dnnl_dim_t>(query_length),
        static_cast<dnnl_dim_t>(value_dimension)};
    const dnnl::memory::dims attended_strides{
        static_cast<dnnl_dim_t>(query_length * value_dimension),
        static_cast<dnnl_dim_t>(value_dimension), 1};
    const auto value_source_desc = dnnl::memory::desc(
        value_source_dims, dnnl::memory::data_type::bf16,
        value_source_strides);
    const auto attended_desc = dnnl::memory::desc(
        attended_dims, dnnl::memory::data_type::bf16,
        attended_strides);
    std::vector<llaisys::bf16_t> attended(
        heads * query_length * value_dimension);
    const dnnl::matmul::primitive_desc value_primitive_desc(
        engine, score_desc, value_source_desc, attended_desc);
    dnnl::matmul(value_primitive_desc)
        .execute(
            stream,
            {{DNNL_ARG_SRC,
                 dnnl::memory(
                     score_desc, engine, probabilities.data())},
             {DNNL_ARG_WEIGHTS,
                 dnnl::memory(value_source_desc, engine, values.data())},
             {DNNL_ARG_DST,
                 dnnl::memory(attended_desc, engine, attended.data())}});
    stream.wait();
    for (size_t token = 0; token < query_length; ++token) {
        for (size_t head = 0; head < heads; ++head) {
            std::copy_n(
                attended.data() +
                    (head * query_length + token) * value_dimension,
                value_dimension,
                destination +
                    (token * heads + head) * value_dimension);
        }
    }
}
#endif

__attribute__((target("avx512f,avx512dq,avx512vl,fma")))
static void qwen2_linear_bf16(llaisysTensor_t out,
                              llaisysTensor_t input,
                              llaisysTensor_t weight,
                              llaisysTensor_t bias,
                              size_t rows,
                              size_t input_size,
                              size_t output_size) {
    auto *destination =
        reinterpret_cast<llaisys::bf16_t *>(tensorGetData(out));
    const auto *source =
        reinterpret_cast<const llaisys::bf16_t *>(tensorGetData(input));
    const auto *matrix =
        reinterpret_cast<const llaisys::bf16_t *>(tensorGetData(weight));
#ifdef LLAISYS_USE_DNNL
    static const dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    static dnnl::stream stream(engine);
    const dnnl::memory::dims source_dims{
        static_cast<dnnl_dim_t>(rows),
        static_cast<dnnl_dim_t>(input_size)};
    const dnnl::memory::dims source_strides{
        static_cast<dnnl_dim_t>(input_size), 1};
    const dnnl::memory::dims weight_dims{
        static_cast<dnnl_dim_t>(input_size),
        static_cast<dnnl_dim_t>(output_size)};
    const dnnl::memory::dims weight_strides{
        1, static_cast<dnnl_dim_t>(input_size)};
    const dnnl::memory::dims output_dims{
        static_cast<dnnl_dim_t>(rows),
        static_cast<dnnl_dim_t>(output_size)};
    const dnnl::memory::dims output_strides{
        static_cast<dnnl_dim_t>(output_size), 1};
    const auto source_desc = dnnl::memory::desc(
        source_dims, dnnl::memory::data_type::bf16, source_strides);
    const auto weight_desc = dnnl::memory::desc(
        weight_dims, dnnl::memory::data_type::bf16, weight_strides);
    const auto output_desc = dnnl::memory::desc(
        output_dims, dnnl::memory::data_type::bf16, output_strides);
    const auto source_memory = dnnl::memory(
        source_desc, engine, const_cast<llaisys::bf16_t *>(source));
    const auto weight_memory = dnnl::memory(
        weight_desc, engine, const_cast<llaisys::bf16_t *>(matrix));
    const auto output_memory =
        dnnl::memory(output_desc, engine, destination);
    std::unordered_map<int, dnnl::memory> arguments{
        {DNNL_ARG_SRC, source_memory},
        {DNNL_ARG_WEIGHTS, weight_memory},
        {DNNL_ARG_DST, output_memory}};
    dnnl::primitive_attr attribute;
    if (bias) {
        const auto *bias_data =
            reinterpret_cast<const llaisys::bf16_t *>(tensorGetData(bias));
        for (size_t row = 0; row < rows; ++row) {
            std::copy_n(
                bias_data, output_size, destination + row * output_size);
        }
        dnnl::post_ops post_ops;
        post_ops.append_sum();
        attribute.set_post_ops(post_ops);
    }
    const dnnl::matmul::primitive_desc descriptor(
        engine, source_desc, weight_desc, output_desc, attribute);
    dnnl::matmul(descriptor).execute(stream, arguments);
    stream.wait();
    static bool reported = false;
    if (!reported) {
        std::fprintf(
            stderr, "[qwen2] fixed oneDNN 3.4.2 bf16 matmul: active\n");
        reported = true;
    }
    return;
#else
    static bool reported = false;
    if (!reported) {
        std::fprintf(
            stderr, "[qwen2] fixed oneDNN unavailable; AVX fallback\n");
        reported = true;
    }
#endif
    {
        (void)destination;
        (void)source;
        (void)matrix;
    }
    for (size_t row = 0; row < rows; ++row) {
        for (size_t output_index = 0; output_index < output_size;
             ++output_index) {
            const auto *source_row = source + row * input_size;
            const auto *matrix_row = matrix + output_index * input_size;
            __m512 accumulator = _mm512_setzero_ps();
            size_t input_index = 0;
            for (; input_index + 16 <= input_size; input_index += 16) {
                const __m256i source16 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(
                        source_row + input_index));
                const __m256i matrix16 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(
                        matrix_row + input_index));
                const __m512 source32 = _mm512_castsi512_ps(
                    _mm512_slli_epi32(
                        _mm512_cvtepu16_epi32(source16), 16));
                const __m512 matrix32 = _mm512_castsi512_ps(
                    _mm512_slli_epi32(
                        _mm512_cvtepu16_epi32(matrix16), 16));
                accumulator =
                    _mm512_fmadd_ps(source32, matrix32, accumulator);
            }
            float sum = _mm512_reduce_add_ps(accumulator);
            if (input_index < input_size) {
                for (; input_index < input_size; ++input_index) {
                    sum += llaisys::utils::cast<float>(
                               source_row[input_index]) *
                           llaisys::utils::cast<float>(
                               matrix_row[input_index]);
                }
            }
            if (bias) {
                sum += llaisys::utils::cast<float>(
                    reinterpret_cast<const llaisys::bf16_t *>(
                        tensorGetData(bias))[output_index]);
            }
            destination[row * output_size + output_index] =
                llaisys::utils::cast<llaisys::bf16_t>(sum);
        }
    }
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

    try {
        weights_alloc(m);
        kv_alloc(m);
        return m;
    } catch (const std::exception &) {
        kv_free(m);
        weights_free_and_destroy_tensors(m);
        delete m;
        return nullptr;
    }
}

__export void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }
    trace_clear(model);
    kv_free(model);
    weights_free_and_destroy_tensors(model);
    delete model;
}

__export const LlaisysQwen2Trace *
llaisysQwen2ModelTrace(LlaisysQwen2Model *model) {
    return model ? &model->trace : nullptr;
}

__export LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
    if (!model) {
        return nullptr;
    }
    return &model->w;
}

static int64_t qwen2_model_infer_impl(
    LlaisysQwen2Model *model,
    const int64_t *token_ids,
    size_t ntoken,
    bool use_cache) {
    // ---- switches ----
    static const bool DEBUG_LAYERS = false;   // 逐层日志（默认关）
    static const bool STAGE_MARKS = false;     // 阶段打点（默认开，用于定位卡住）
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
        if ((!use_cache && ntoken > model->meta.maxseq) ||
            (use_cache && ntoken > model->meta.maxseq - model->cached_len)) {
            std::fprintf(stderr, "[infer] ntoken too long: ntoken=%zu maxseq=%zu\n", ntoken, model->meta.maxseq);
            return -3;
        }
        const auto &meta = model->meta;
        const int devid = pick_device_id(model);
        size_t diagnostic_layer = 0;
        if (const char *configured =
                std::getenv("LLAISYS_QWEN2_TRACE_LAYER")) {
            char *end = nullptr;
            const auto parsed = std::strtoull(configured, &end, 10);
            if (end && *end == '\0' && parsed < meta.nlayer) {
                diagnostic_layer = static_cast<size_t>(parsed);
            }
        }
        auto linear = [&](llaisysTensor_t out,
                          llaisysTensor_t input,
                          llaisysTensor_t weight,
                          llaisysTensor_t bias,
                          size_t rows,
                          size_t input_size,
                          size_t output_size) {
            if (model->device == LLAISYS_DEVICE_CPU &&
                meta.dtype == LLAISYS_DTYPE_BF16) {
                qwen2_linear_bf16(
                    out, input, weight, bias,
                    rows, input_size, output_size);
            } else {
                llaisysLinear(out, input, weight, bias);
                require_op_success("linear");
            }
        };
        trace_clear(model);
        model->trace.attention_out = new llaisysTensor_t[meta.nlayer]();
        model->trace.layer_output = new llaisysTensor_t[meta.nlayer]();

        std::fprintf(stderr,
                     "[infer] device=%d devid=%d ntoken=%zu cache_mode=%d dtype=%d nlayer=%zu hs=%zu nh=%zu nkvh=%zu dh=%zu di=%zu voc=%zu maxseq=%zu\n",
                     (int)model->device, devid, ntoken, use_cache ? 1 : 0, (int)meta.dtype, meta.nlayer, meta.hs, meta.nh,
                     meta.nkvh, meta.dh, meta.di, meta.voc, meta.maxseq);
        std::fflush(stderr);

        // M4 no-cache calls always recompute their complete supplied prefix.
        // M5 cached calls accept only the new chunk and use an explicit cursor.
        const size_t start = use_cache ? model->cached_len : 0;
        const size_t new_len = ntoken;
        const size_t total_len = start + new_len;
        std::fprintf(stderr, "[infer] start=%zu new_len=%zu\n", start, new_len);
        std::fflush(stderr);
        if (new_len == 0) {
            return -4;
        }

        const float attn_scale = 1.0f / std::sqrt((float)meta.dh);

        // token ids new: [new_len] int64
        MARK("alloc/load tokens");
        TensorGuard t_tok(make_tensor_1d(new_len, LLAISYS_DTYPE_I64, model->device, devid));
        tensorLoad(t_tok.t, token_ids);

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
        require_op_success("embedding");
        model->trace.embedding = trace_copy(x.t);

        MARK("before blocks");

        // blocks
        for (size_t l = 0; l < meta.nlayer; ++l) {
            MARK_L(l, "begin");

            if (l == 0) {
                MARK("L0: attn rmsnorm");
            }
            TensorGuard xn(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
            llaisysRmsNorm(xn.t, x.t, model->w.attn_norm_w[l], meta.epsilon);
            require_op_success("input RMSNorm");
            if (l == diagnostic_layer) {
                model->trace.diagnostic_attn_norm = trace_copy(xn.t);
            }

            if (l == 0) {
                MARK("L0: qkv linear");
            }
            TensorGuard q2(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
            TensorGuard k2(make_tensor_2d(new_len, meta.nkvh * meta.dh, meta.dtype, model->device, devid));
            TensorGuard v2(make_tensor_2d(new_len, meta.nkvh * meta.dh, meta.dtype, model->device, devid));
            linear(q2.t, xn.t, model->w.attn_q_w[l],
                   model->w.attn_q_b ? model->w.attn_q_b[l] : nullptr,
                   new_len, meta.hs, meta.hs);
            linear(k2.t, xn.t, model->w.attn_k_w[l],
                   model->w.attn_k_b ? model->w.attn_k_b[l] : nullptr,
                   new_len, meta.hs, meta.nkvh * meta.dh);
            linear(v2.t, xn.t, model->w.attn_v_w[l],
                   model->w.attn_v_b ? model->w.attn_v_b[l] : nullptr,
                   new_len, meta.hs, meta.nkvh * meta.dh);
            if (l == diagnostic_layer) {
                model->trace.diagnostic_q = trace_copy(q2.t);
                model->trace.diagnostic_k = trace_copy(k2.t);
                model->trace.diagnostic_v = trace_copy(v2.t);
            }

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
            if (model->device == LLAISYS_DEVICE_CPU &&
                meta.dtype == LLAISYS_DTYPE_BF16) {
                qwen2_rope_bf16(
                    q_rope.t, q3.t, t_pos.t, meta.theta,
                    new_len, meta.nh, meta.dh);
                qwen2_rope_bf16(
                    k_rope.t, k3.t, t_pos.t, meta.theta,
                    new_len, meta.nkvh, meta.dh);
            } else {
                llaisysROPE(q_rope.t, q3.t, t_pos.t, meta.theta);
                llaisysROPE(k_rope.t, k3.t, t_pos.t, meta.theta);
                require_op_success("RoPE");
            }
            if (l == diagnostic_layer) {
                model->trace.diagnostic_q_rope = trace_copy(q_rope.t);
                model->trace.diagnostic_k_rope = trace_copy(k_rope.t);
            }

            TensorGuard k_total;
            TensorGuard v_total;
            llaisysTensor_t attention_k = k_rope.t;
            llaisysTensor_t attention_v = v3.t;
            if (use_cache) {
                const size_t row_elements = meta.nkvh * meta.dh;
                const size_t row_bytes =
                    row_elements * llaisys::utils::dsize(meta.dtype);
                auto *cache_k = reinterpret_cast<std::byte *>(
                    tensorGetData(model->k_cache[l]));
                auto *cache_v = reinterpret_cast<std::byte *>(
                    tensorGetData(model->v_cache[l]));
                const auto *new_k = reinterpret_cast<const std::byte *>(
                    tensorGetData(k_rope.t));
                const auto *new_v = reinterpret_cast<const std::byte *>(
                    tensorGetData(v3.t));
                if (!cache_k || !cache_v || !new_k || !new_v) {
                    throw std::runtime_error("KV cache data unavailable");
                }
                const auto copy_kind = model->device == LLAISYS_DEVICE_CPU
                    ? LLAISYS_MEMCPY_H2H
                    : LLAISYS_MEMCPY_D2D;
                copy_tensor_bytes(
                    model, cache_k + start * row_bytes, new_k,
                    new_len * row_bytes, copy_kind);
                copy_tensor_bytes(
                    model, cache_v + start * row_bytes, new_v,
                    new_len * row_bytes, copy_kind);
                k_total = TensorGuard(tensorSlice(
                    model->k_cache[l], 0, 0, total_len));
                v_total = TensorGuard(tensorSlice(
                    model->v_cache[l], 0, 0, total_len));
                if (!k_total.t || !v_total.t) {
                    throw std::runtime_error("failed to view Qwen2 KV cache");
                }
                attention_k = k_total.t;
                attention_v = v_total.t;
            }

            if (l == 0) {
                MARK("L0: self_attention");
            }
            // dv 在你的 meta 里通常等于 dh；如果不是，且你的 self_attention 实现很慢，
            // 可临时开 FORCE_DV_EQ_D 做定位。
            const size_t dv = FORCE_DV_EQ_D ? meta.dh : meta.dh;

            TensorGuard attn_val(make_tensor_3d(new_len, meta.nh, dv, meta.dtype, model->device, devid));
            if (l == 0) {
                dump_shape("q_rope", q_rope.t);
                dump_shape("k_total", attention_k);
                dump_shape("v_total", attention_v);
                dump_shape("attn_val(out, pre)", attn_val.t);
            }

            if (model->device == LLAISYS_DEVICE_CPU &&
                meta.dtype == LLAISYS_DTYPE_BF16) {
#ifdef LLAISYS_USE_DNNL
                qwen2_attention_bf16(
                    attn_val.t, q_rope.t, attention_k, attention_v,
                    new_len, total_len, meta.nh, meta.nkvh, meta.dh, dv,
                    attn_scale,
                    l == diagnostic_layer
                        ? &model->trace.diagnostic_attention_scores
                        : nullptr,
                    l == diagnostic_layer
                        ? &model->trace.diagnostic_attention_probabilities
                        : nullptr);
#else
                llaisysSelfAttention(
                    attn_val.t, q_rope.t, attention_k, attention_v,
                    attn_scale);
                require_op_success("self attention");
#endif
            } else {
                llaisysSelfAttention(
                    attn_val.t, q_rope.t, attention_k, attention_v,
                    attn_scale);
                require_op_success("self attention");
            }
            if (l == diagnostic_layer) {
                model->trace.diagnostic_attn_value = trace_copy(attn_val.t);
            }

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
            linear(attn_out.t, attn_2d.t, model->w.attn_o_w[l], nullptr,
                   new_len, meta.hs, meta.hs);
            model->trace.attention_out[l] = trace_copy(attn_out.t);

            MARK_L(l, "residual add 1");
            TensorGuard x1(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
            llaisysAdd(x1.t, x.t, attn_out.t);
            if (l == diagnostic_layer) {
                model->trace.diagnostic_post_attention = trace_copy(x1.t);
            }

            if (l == 0) {
                MARK("L0: mlp");
            }
            TensorGuard x1n(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
            llaisysRmsNorm(x1n.t, x1.t, model->w.mlp_norm_w[l], meta.epsilon);
            if (l == diagnostic_layer) {
                model->trace.diagnostic_mlp_norm = trace_copy(x1n.t);
            }

            TensorGuard gate(make_tensor_2d(new_len, meta.di, meta.dtype, model->device, devid));
            TensorGuard up(make_tensor_2d(new_len, meta.di, meta.dtype, model->device, devid));
            linear(gate.t, x1n.t, model->w.mlp_gate_w[l], nullptr,
                   new_len, meta.hs, meta.di);
            linear(up.t, x1n.t, model->w.mlp_up_w[l], nullptr,
                   new_len, meta.hs, meta.di);
            if (l == diagnostic_layer) {
                model->trace.diagnostic_gate = trace_copy(gate.t);
                model->trace.diagnostic_up = trace_copy(up.t);
            }

            TensorGuard act(make_tensor_2d(new_len, meta.di, meta.dtype, model->device, devid));
            if (model->device == LLAISYS_DEVICE_CPU &&
                meta.dtype == LLAISYS_DTYPE_BF16) {
                qwen2_swiglu_bf16(
                    act.t, gate.t, up.t, new_len * meta.di);
            } else {
                llaisysSwiGLU(act.t, gate.t, up.t);
                require_op_success("swiglu");
            }
            if (l == diagnostic_layer) {
                model->trace.diagnostic_activation = trace_copy(act.t);
            }

            TensorGuard mlp_out(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
            linear(mlp_out.t, act.t, model->w.mlp_down_w[l], nullptr,
                   new_len, meta.di, meta.hs);
            if (l == diagnostic_layer) {
                model->trace.diagnostic_mlp_out = trace_copy(mlp_out.t);
            }

            TensorGuard x2(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
            llaisysAdd(x2.t, x1.t, mlp_out.t);
            require_op_success("MLP residual");

            x = std::move(x2);
            model->trace.layer_output[l] = trace_copy(x.t);
            MARK_L(l, "end");
        }

        MARK("after blocks");

        MARK("final norm");
        TensorGuard xnorm(make_tensor_2d(new_len, meta.hs, meta.dtype, model->device, devid));
        llaisysRmsNorm(xnorm.t, x.t, model->w.out_norm_w, meta.epsilon);
        require_op_success("final RMSNorm");
        model->trace.final_norm = trace_copy(xnorm.t);

        MARK("logits linear");
        TensorGuard logits(make_tensor_2d(new_len, meta.voc, meta.dtype, model->device, devid));
        linear(logits.t, xnorm.t, model->w.out_embed, nullptr,
               new_len, meta.hs, meta.voc);
        model->trace.logits = trace_copy(logits.t);

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
        int64_t out = 0;
        copy_tensor_bytes(
            model, &out, p, sizeof(out),
            model->device == LLAISYS_DEVICE_CPU
                ? LLAISYS_MEMCPY_H2H
                : LLAISYS_MEMCPY_D2H);
        model->trace.greedy_token = out;

        if (use_cache) {
            model->cached_len = total_len;
            model->cached_tokens.insert(
                model->cached_tokens.end(), token_ids, token_ids + ntoken);
        }

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

__export int64_t llaisysQwen2ModelInfer(
    LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    return qwen2_model_infer_impl(model, token_ids, ntoken, false);
}

__export int64_t llaisysQwen2ModelInferCached(
    LlaisysQwen2Model *model, llaisysTensor_t token_ids) {
    if (!model || !token_ids) {
        return -1;
    }
    if (tensorGetNdim(token_ids) != 1) {
        return -4;
    }
    if (tensorGetDataType(token_ids) != LLAISYS_DTYPE_I64) {
        return -5;
    }
    if (tensorGetDeviceType(token_ids) != model->device ||
        tensorGetDeviceId(token_ids) != pick_device_id(model)) {
        return -6;
    }
    size_t shape[1]{};
    tensorGetShape(token_ids, shape);
    if (shape[0] == 0) {
        return -7;
    }
    auto *data = reinterpret_cast<const int64_t *>(tensorGetData(token_ids));
    if (!data) {
        return -1;
    }
    if (model->device == LLAISYS_DEVICE_CPU) {
        return qwen2_model_infer_impl(model, data, shape[0], true);
    }
    try {
        std::vector<int64_t> host_tokens(shape[0]);
        copy_tensor_bytes(
            model, host_tokens.data(), data,
            shape[0] * sizeof(int64_t), LLAISYS_MEMCPY_D2H);
        return qwen2_model_infer_impl(
            model, host_tokens.data(), host_tokens.size(), true);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[infer-cached] exception caught: %s\n", e.what());
        return -999;
    }
}

__export int llaisysQwen2ModelResetCache(LlaisysQwen2Model *model) {
    if (!model) {
        return -1;
    }
    model->cached_len = 0;
    model->cached_tokens.clear();
    return 0;
}

__export size_t llaisysQwen2ModelCacheCursor(
    const LlaisysQwen2Model *model) {
    return model ? model->cached_len : 0;
}

__export size_t llaisysQwen2ModelCacheCapacity(
    const LlaisysQwen2Model *model) {
    return model ? model->meta.maxseq : 0;
}

__export size_t llaisysQwen2ModelCacheAllocatedBytes(
    const LlaisysQwen2Model *model) {
    try {
        return kv_payload_bytes(model);
    } catch (...) {
        return 0;
    }
}

__export uintptr_t llaisysQwen2ModelCacheAddress(
    const LlaisysQwen2Model *model, size_t layer, int value_cache) {
    if (!model || layer >= model->meta.nlayer ||
        (value_cache != 0 && value_cache != 1)) {
        return 0;
    }
    const auto tensor =
        value_cache == 0 ? model->k_cache[layer] : model->v_cache[layer];
    return reinterpret_cast<uintptr_t>(tensorGetData(tensor));
}

} // extern "C"
