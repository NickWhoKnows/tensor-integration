#include "tllm/ops/attn.h"

#include "tllm/ops/norm.h"
#include "tllm/ops/rope.h"
#include "tllm/runtime/kv_cache.h"

#include <cmath>

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

    ggml_tensor *past = ggml_view_3d(ctx, cache, cache->ne[0], cache->ne[1], n_past, cache->nb[1],
                                     cache->nb[2], 0);
    return ggml_concat(ctx, past, cur, 2);
}

ggml_tensor *store_in_cache(ggml_context *ctx, ggml_tensor *cache, ggml_tensor *src,
                            const int head_dim, const int n_head_kv, const int n_tokens,
                            const size_t slot_offset)
{
    ggml_tensor *slot = ggml_view_3d(ctx, cache, head_dim, n_head_kv, n_tokens, cache->nb[1],
                                     cache->nb[2], slot_offset);
    ggml_tensor *cpy = ggml_cpy(ctx, src, slot);
    ggml_set_output(cpy);
    return cpy;
}

} // namespace

ggml_tensor *attention(ggml_context *ctx, ggml_tensor *x, const AttnWeights &weights,
                       const model::Config &config, ggml_tensor *positions,
                       runtime::LayerKvCache *cache, const int position_offset)
{
    const int head_dim = config.n_embd / config.n_head;
    const int n_tokens = static_cast<int>(x->ne[1]);
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    ggml_tensor *cur = rms_norm(ctx, x, weights.norm, config.rms_norm_eps);
    ggml_tensor *q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weights.q, cur), head_dim, config.n_head,
                                     n_tokens);
    ggml_tensor *k_new =
        ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weights.k, cur), head_dim, config.n_head_kv, n_tokens);
    ggml_tensor *v_new =
        ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weights.v, cur), head_dim, config.n_head_kv, n_tokens);

    q = apply_rope(ctx, q, positions, config);
    k_new = apply_rope(ctx, k_new, positions, config);

    ggml_tensor *k = k_new;
    ggml_tensor *v = v_new;
    if (cache != nullptr)
    {
        const size_t slot_offset = static_cast<size_t>(position_offset) * cache->k->nb[2];
        cache->last_k =            store_in_cache(ctx, cache->k, k_new, head_dim, config.n_head_kv, n_tokens, slot_offset);
        cache->last_v =            store_in_cache(ctx, cache->v, v_new, head_dim, config.n_head_kv, n_tokens, slot_offset);
        k =                        concat_past(ctx, cache->k, k_new, position_offset);
        v =                        concat_past(ctx, cache->v, v_new, position_offset);
    }

    const int n_kv = cache != nullptr ? position_offset + n_tokens : n_tokens;

    ggml_tensor *kq = ggml_mul_mat(ctx, ggml_permute(ctx, k, 0, 2, 1, 3), ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_soft_max(ctx, ggml_diag_mask_inf(ctx, ggml_scale(ctx, kq, kq_scale), position_offset));

    ggml_tensor *v_perm = ggml_cont_3d(ctx, ggml_permute(ctx, v, 1, 2, 0, 3), n_kv, head_dim, config.n_head_kv);
    cur =                 ggml_cont_2d(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, v_perm, kq), 0, 2, 1, 3), config.n_embd,
                       n_tokens);

    return ggml_mul_mat(ctx, weights.output, cur);
}

} // namespace tllm::ops
