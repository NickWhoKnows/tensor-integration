#include "tllm/ops/layer.h"

#include "tllm/model/weights.h"

#include <cmath>
#include <stdexcept>

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

void KvCache::init(ggml_backend_t backend, const model::Config &config, const int max_ctx)
{
    if (backend == nullptr || max_ctx <= 0)
    {
        throw std::runtime_error("invalid kv cache init");
    }

    const int head_dim = config.n_embd / config.n_head;
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
        layer.k = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, head_dim, config.n_head_kv, max_ctx_);
        layer.v = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, head_dim, config.n_head_kv, max_ctx_);
    }

    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
    if (buffer_ == nullptr)
    {
        throw std::runtime_error("failed to allocate kv cache tensors");
    }

    ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
}

void KvCache::advance(const int n_tokens)
{
    if (n_past_ + n_tokens > max_ctx_)
    {
        throw std::runtime_error("kv cache overflow");
    }
    n_past_ += n_tokens;
}

} // namespace tllm::runtime

namespace tllm::ops
{
namespace
{

ggml_tensor *concat_past(ggml_context *ctx, ggml_tensor *cache, ggml_tensor *cur, const int n_past)
{
    if (n_past <= 0)
    {
        return cur;
    }

    ggml_tensor *past =
        ggml_view_3d(ctx, cache, cache->ne[0], cache->ne[1], n_past, cache->nb[1], cache->nb[2], 0);
    return ggml_concat(ctx, past, cur, 2);
}

ggml_tensor *store_in_cache(ggml_context *ctx, ggml_tensor *cache, ggml_tensor *src, const int head_dim,
                            const int n_head_kv, const int n_tokens, const size_t slot_offset)
{
    ggml_tensor *slot =
        ggml_view_3d(ctx, cache, head_dim, n_head_kv, n_tokens, cache->nb[1], cache->nb[2], slot_offset);
    ggml_tensor *cpy = ggml_cpy(ctx, src, slot);
    ggml_set_output(cpy);
    return cpy;
}

} // namespace

ggml_tensor *rms_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *weight, const float eps)
{
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

ggml_tensor *embed_tokens(ggml_context *ctx, ggml_tensor *token_embd, const std::vector<int32_t> &tokens)
{
    ggml_tensor *ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(tokens.size()));
    ggml_set_name(ids, kInputTokens);
    ggml_set_input(ids);
    return ggml_get_rows(ctx, token_embd, ids);
}

ggml_tensor *make_positions_input(ggml_context *ctx, const int n_tokens)
{
    ggml_tensor *positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, kRopePositions);
    ggml_set_input(positions);
    return positions;
}

ggml_tensor *apply_rope(ggml_context *ctx, ggml_tensor *qkv, ggml_tensor *positions,
                        const model::Config &config)
{
    return ggml_rope_ext(ctx, qkv, positions, config.rope_freqs, config.rope_dimension_count,
                         GGML_ROPE_TYPE_NORMAL, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f,
                         0.0f);
}

ggml_tensor *ffn(ggml_context *ctx, ggml_tensor *x, const FfnWeights &weights, const float eps)
{
    ggml_tensor *cur = rms_norm(ctx, x, weights.norm, eps);
    ggml_tensor *gate_out = ggml_mul_mat(ctx, weights.gate, cur);
    ggml_tensor *up_out = ggml_mul_mat(ctx, weights.up, cur);
    ggml_tensor *hidden = ggml_mul(ctx, up_out, ggml_silu(ctx, gate_out));
    return ggml_mul_mat(ctx, weights.down, hidden);
}

ggml_tensor *attention(ggml_context *ctx, ggml_tensor *x, const AttnWeights &weights, const model::Config &config,
                       ggml_tensor *positions, runtime::LayerKvCache *cache, const int position_offset)
{
    const int head_dim = config.n_embd / config.n_head;
    const int n_tokens = static_cast<int>(x->ne[1]);
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    ggml_tensor *cur = rms_norm(ctx, x, weights.norm, config.rms_norm_eps);
    ggml_tensor *q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weights.q, cur), head_dim, config.n_head, n_tokens);
    ggml_tensor *k_new = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weights.k, cur), head_dim, config.n_head_kv, n_tokens);
    ggml_tensor *v_new = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weights.v, cur), head_dim, config.n_head_kv, n_tokens);

    q = apply_rope(ctx, q, positions, config);
    k_new = apply_rope(ctx, k_new, positions, config);

    ggml_tensor *k = k_new;
    ggml_tensor *v = v_new;
    if (cache != nullptr)
    {
        const size_t slot_offset = static_cast<size_t>(position_offset) * cache->k->nb[2];
        cache->last_k = store_in_cache(ctx, cache->k, k_new, head_dim, config.n_head_kv, n_tokens, slot_offset);
        cache->last_v = store_in_cache(ctx, cache->v, v_new, head_dim, config.n_head_kv, n_tokens, slot_offset);
        k = concat_past(ctx, cache->k, k_new, position_offset);
        v = concat_past(ctx, cache->v, v_new, position_offset);
    }

    const int n_kv = cache != nullptr ? position_offset + n_tokens : n_tokens;

    ggml_tensor *kq = ggml_mul_mat(ctx, ggml_permute(ctx, k, 0, 2, 1, 3), ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_soft_max(ctx, ggml_diag_mask_inf(ctx, ggml_scale(ctx, kq, kq_scale), position_offset));

    ggml_tensor *v_perm = ggml_cont_3d(ctx, ggml_permute(ctx, v, 1, 2, 0, 3), n_kv, head_dim, config.n_head_kv);
    cur = ggml_cont_2d(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, v_perm, kq), 0, 2, 1, 3), config.n_embd, n_tokens);

    return ggml_mul_mat(ctx, weights.output, cur);
}

Layer::Layer(const int index, ggml_context * /*ctx*/, const model::Config &config, gguf::Loader &loader)
    : config_(config)
{
    const model::BlockWeights names = model::block_weights(index);
    attn_ = AttnWeights{
        .norm = loader.wrap_tensor(names.attn_norm),
        .q = loader.wrap_tensor(names.attn_q),
        .k = loader.wrap_tensor(names.attn_k),
        .v = loader.wrap_tensor(names.attn_v),
        .output = loader.wrap_tensor(names.attn_output),
    };
    ffn_ = FfnWeights{
        .norm = loader.wrap_tensor(names.ffn_norm),
        .gate = loader.wrap_tensor(names.ffn_gate),
        .up = loader.wrap_tensor(names.ffn_up),
        .down = loader.wrap_tensor(names.ffn_down),
    };
}

ggml_tensor *Layer::block_transformer(ggml_context *ctx, ggml_tensor *x, ggml_tensor *positions,
                                      runtime::LayerKvCache *cache, const int position_offset) const
{
    ggml_tensor *attn_out = attention(ctx, x, attn_, config_, positions, cache, position_offset);
    ggml_tensor *x_attn = ggml_add(ctx, x, attn_out);
    ggml_tensor *ffn_out = ffn(ctx, x_attn, ffn_, config_.rms_norm_eps);
    return ggml_add(ctx, x_attn, ffn_out);
}

} // namespace tllm::ops
