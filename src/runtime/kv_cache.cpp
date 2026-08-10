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
        buffer_ = nullptr;
    }

    if (ctx_ != nullptr)
    {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
}

void KvCache::init(const Backend &backend, const model::Config &config, const int max_ctx)
{
    if (max_ctx <= 0)
    {
        throw std::runtime_error("kv cache max_ctx must be positive");
    }

    head_dim_ = config.n_embd / config.n_head;
    n_head_kv_ = config.n_head_kv;
    max_ctx_ = max_ctx;
    n_past_ = 0;

    const size_t n_tensors = static_cast<size_t>(config.n_layer) * 2;
    ggml_init_params params{
        .mem_size = ggml_tensor_overhead() * n_tensors,
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
        layer.k = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, head_dim_, n_head_kv_, max_ctx_);
        layer.v = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, head_dim_, n_head_kv_, max_ctx_);
        layer.last_k = nullptr;
        layer.last_v = nullptr;
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

void KvCache::commit_layer(const int layer_index, const int head_dim, const int n_head_kv,
                           const Backend *backend)
{
    LayerKvCache &entry = layers_.at(static_cast<size_t>(layer_index));
    if (entry.last_k == nullptr || entry.last_v == nullptr || backend == nullptr)
    {
        return;
    }

    const int n_tokens = static_cast<int>(entry.last_k->ne[2]);
    if (n_past_ + n_tokens > max_ctx_)
    {
        throw std::runtime_error("kv cache overflow");
    }

    const size_t row_elems = static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv);
    const size_t nbytes = row_elems * static_cast<size_t>(n_tokens) * sizeof(float);
    const size_t dst_offset = static_cast<size_t>(n_past_) * row_elems * sizeof(float);

    std::vector<float> staging(row_elems * static_cast<size_t>(n_tokens));
    backend->tensor_get(entry.last_k, staging.data(), 0, nbytes);
    backend->tensor_set(entry.k, staging.data(), dst_offset, nbytes);
    backend->tensor_get(entry.last_v, staging.data(), 0, nbytes);
    backend->tensor_set(entry.v, staging.data(), dst_offset, nbytes);
}

} // namespace tllm::runtime
