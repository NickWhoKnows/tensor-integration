#include "tllm/ops/attn.h"

#include "tllm/ops/norm.h"
#include "tllm/ops/rope.h"
#include "tllm/runtime/kv_cache.h"

#include <cmath>

namespace tllm::ops
{

namespace
{

ggml_tensor *concat_past_kv(ggml_context *ctx, ggml_tensor *cache_kv, ggml_tensor *new_kv,
                            const int n_past)
{
    if (n_past <= 0)
    {
        return new_kv;
    }

    ggml_tensor *past = ggml_view_3d(ctx, cache_kv, cache_kv->ne[0], cache_kv->ne[1], n_past,
                                     cache_kv->nb[1], cache_kv->nb[2], 0);
    return ggml_concat(ctx, past, new_kv, 2);
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

    ggml_tensor *q_cur = ggml_mul_mat(ctx, weights.q, cur);
    ggml_tensor *k_cur = ggml_mul_mat(ctx, weights.k, cur);
    ggml_tensor *v_cur = ggml_mul_mat(ctx, weights.v, cur);

    ggml_tensor *q = ggml_reshape_3d(ctx, q_cur, head_dim, config.n_head, n_tokens);
    ggml_tensor *k_new = ggml_reshape_3d(ctx, k_cur, head_dim, config.n_head_kv, n_tokens);
    ggml_tensor *v_new = ggml_reshape_3d(ctx, v_cur, head_dim, config.n_head_kv, n_tokens);

    q = apply_rope(ctx, q, positions, config);
    k_new = apply_rope(ctx, k_new, positions, config);
    v_new = apply_rope(ctx, v_new, positions, config);

    ggml_tensor *k = k_new;
    ggml_tensor *v = v_new;
    if (cache != nullptr)
    {
        cache->last_k = k_new;
        cache->last_v = v_new;
        ggml_set_output(k_new);
        ggml_set_output(v_new);

        k = concat_past_kv(ctx, cache->k, k_new, position_offset);
        v = concat_past_kv(ctx, cache->v, v_new, position_offset);
    }

    const int n_kv = cache != nullptr ? position_offset + n_tokens : n_tokens;

    ggml_tensor *q_perm = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor *k_perm = ggml_permute(ctx, k, 0, 2, 1, 3);

    ggml_tensor *kq = ggml_mul_mat(ctx, k_perm, q_perm);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_scale(ctx, kq, kq_scale);
    kq = ggml_diag_mask_inf(ctx, kq, position_offset);
    kq = ggml_soft_max(ctx, kq);

    ggml_tensor *v_perm = ggml_cont_3d(ctx, ggml_permute(ctx, v, 1, 2, 0, 3), n_kv, head_dim,
                                       config.n_head_kv);

    ggml_tensor *kqv = ggml_mul_mat(ctx, v_perm, kq);
    ggml_tensor *kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    cur = ggml_cont_2d(ctx, kqv_merged, config.n_embd, n_tokens);

    return ggml_mul_mat(ctx, weights.output, cur);
}

} // namespace tllm::ops
