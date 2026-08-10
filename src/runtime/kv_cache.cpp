#include "tllm/runtime/kv_cache.h"

#include "tllm/runtime/backend.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <stdexcept>

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

    buffer_ = backend.alloc_ctx_tensors(ctx_);
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

} // namespace tllm::runtime
