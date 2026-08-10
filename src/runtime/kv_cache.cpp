#include "tllm/runtime/kv_cache.h"

#include "tllm/runtime/backend.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <stdexcept>
#include <vector>

namespace tllm::runtime
{

KvCache::~KvCache()
{
    if (buffer_ != nullptr)
    {
        ggml_backend_buffer_free(buffer_);
    }
    if (ctx_ != nullptr)
    {
        ggml_free(ctx_);
    }
}

void KvCache::init(const Backend &backend, const model::Config &config, const int max_ctx)
{
    if (max_ctx <= 0)
    {
        throw std::runtime_error("kv cache max_ctx must be positive");
    }

    const int head_dim = config.n_embd / config.n_head;
    const int n_head_kv = config.n_head_kv;
    max_ctx_ = max_ctx;
    n_past_ = 0;

    ggml_init_params params{
        .mem_size = ggml_tensor_overhead() * static_cast<size_t>(config.n_layer) * 2,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };

    ctx_ = ggml_init(params);
    if (ctx_ == nullptr)
    {
        throw std::runtime_error("failed to initialize kv cache context");
    }

    layers_.resize(static_cast<size_t>(config.n_layer));
    for (int i = 0; i < config.n_layer; ++i)
    {
        LayerKvCache &layer = layers_[static_cast<size_t>(i)];
        layer.k = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, head_dim, n_head_kv, max_ctx_);
        layer.v = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, head_dim, n_head_kv, max_ctx_);
    }

    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend.cpu_handle());
    if (buffer_ == nullptr)
    {
        throw std::runtime_error("failed to allocate kv cache tensors");
    }

    ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
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

void KvCache::commit_layer(LayerKvCache &layer, const Backend *backend)
{
    if (layer.last_k == nullptr || layer.last_v == nullptr || backend == nullptr)
    {
        return;
    }

    const int n_tokens = static_cast<int>(layer.last_k->ne[2]);
    const size_t nbytes = static_cast<size_t>(n_tokens) * layer.k->nb[2];
    const size_t dst_offset = static_cast<size_t>(n_past_) * layer.k->nb[2];

    if (layer.k->data != nullptr && ggml_backend_buffer_is_host(layer.k->buffer))
    {
        backend->tensor_get(layer.last_k, static_cast<char *>(layer.k->data) + dst_offset, 0, nbytes);
        backend->tensor_get(layer.last_v, static_cast<char *>(layer.v->data) + dst_offset, 0, nbytes);
        return;
    }

    std::vector<float> staging(nbytes / sizeof(float));
    backend->tensor_get(layer.last_k, staging.data(), 0, nbytes);
    backend->tensor_set(layer.k, staging.data(), dst_offset, nbytes);
    backend->tensor_get(layer.last_v, staging.data(), 0, nbytes);
    backend->tensor_set(layer.v, staging.data(), dst_offset, nbytes);
}

void KvCache::commit(const Backend *backend, const int n_tokens)
{
    if (n_past_ + n_tokens > max_ctx_)
    {
        throw std::runtime_error("kv cache overflow");
    }

    for (auto &layer : layers_)
    {
        commit_layer(layer, backend);
    }
    n_past_ += n_tokens;
}

} // namespace tllm::runtime
