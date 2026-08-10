#include "tllm/runtime/kv_cache.h"

#include <cstring>
#include <stdexcept>

namespace tllm::runtime
{

void KvCache::init(ggml_context *ctx, const model::Config &config, const int max_ctx)
{
    if (max_ctx <= 0)
    {
        throw std::runtime_error("kv cache max_ctx must be positive");
    }

    head_dim_ = config.n_embd / config.n_head;
    n_head_kv_ = config.n_head_kv;
    max_ctx_ = max_ctx;
    n_past_ = 0;

    layers_.resize(static_cast<size_t>(config.n_layer));
    for (int i = 0; i < config.n_layer; ++i)
    {
        LayerKvCache &layer = layers_[static_cast<size_t>(i)];
        layer.k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim_, n_head_kv_, max_ctx_);
        layer.v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim_, n_head_kv_, max_ctx_);
        layer.last_k = nullptr;
        layer.last_v = nullptr;
    }
}

void KvCache::reset()
{
    n_past_ = 0;
    for (auto &layer : layers_)
    {
        layer.last_k = nullptr;
        layer.last_v = nullptr;
    }
}

void KvCache::commit_layer(const int layer_index, const int head_dim, const int n_head_kv)
{
    LayerKvCache &entry = layers_.at(static_cast<size_t>(layer_index));
    if (entry.last_k == nullptr || entry.last_v == nullptr)
    {
        return;
    }

    const int n_tokens = static_cast<int>(entry.last_k->ne[2]);
    if (n_past_ + n_tokens > max_ctx_)
    {
        throw std::runtime_error("kv cache overflow");
    }

    const size_t row_elems = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv);
    const size_t n_elems = row_elems * static_cast<size_t>(n_tokens);

    const float *k_src = static_cast<const float *>(entry.last_k->data);
    const float *v_src = static_cast<const float *>(entry.last_v->data);
    float *k_dst = static_cast<float *>(entry.k->data) + static_cast<size_t>(n_past_) * row_elems;
    float *v_dst = static_cast<float *>(entry.v->data) + static_cast<size_t>(n_past_) * row_elems;

    std::memcpy(k_dst, k_src, n_elems * sizeof(float));
    std::memcpy(v_dst, v_src, n_elems * sizeof(float));
}

} // namespace tllm::runtime
