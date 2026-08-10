#include "tllm/ops/attn.h"

#include "tllm/ops/norm.h"
#include "tllm/ops/rope.h"

#include <cmath>

namespace tllm::ops
{

ggml_tensor *attention(ggml_context *ctx, ggml_tensor *x, const AttnWeights &weights,
                       const model::Config &config, const int position_offset)
{
    const int head_dim = config.n_embd / config.n_head;
    const int n_tokens = static_cast<int>(x->ne[1]);
    const int n_rep = config.n_head / config.n_head_kv;
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    ggml_tensor *positions = make_positions(ctx, n_tokens, position_offset);
    ggml_tensor *cur = rms_norm(ctx, x, weights.norm, config.rms_norm_eps);

    ggml_tensor *q_cur = ggml_mul_mat(ctx, weights.q, cur);
    ggml_tensor *k_cur = ggml_mul_mat(ctx, weights.k, cur);
    ggml_tensor *v_cur = ggml_mul_mat(ctx, weights.v, cur);

    ggml_tensor *q = ggml_reshape_3d(ctx, q_cur, head_dim, config.n_head, n_tokens);
    ggml_tensor *k = ggml_reshape_3d(ctx, k_cur, head_dim, config.n_head_kv, n_tokens);
    ggml_tensor *v = ggml_reshape_3d(ctx, v_cur, head_dim, config.n_head_kv, n_tokens);

    q = apply_rope(ctx, q, positions, config);
    k = apply_rope(ctx, k, positions, config);

    if (n_rep > 1)
    {
        k = ggml_repeat_4d(ctx, k, k->ne[0], k->ne[1] * n_rep, k->ne[2], k->ne[3]);
        v = ggml_repeat_4d(ctx, v, v->ne[0], v->ne[1] * n_rep, v->ne[2], v->ne[3]);
    }

    ggml_tensor *q_perm = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor *k_perm = ggml_permute(ctx, k, 0, 2, 1, 3);

    ggml_tensor *kq = ggml_mul_mat(ctx, k_perm, q_perm);
    kq = ggml_scale(ctx, kq, kq_scale);
    kq = ggml_diag_mask_inf(ctx,kq,0);
    kq = ggml_soft_max(ctx, kq);

    ggml_tensor *v_perm = ggml_cont_3d(ctx, ggml_permute(ctx, v, 1, 2, 0, 3), n_tokens, head_dim,
                                       config.n_head);

    ggml_tensor *kqv = ggml_mul_mat(ctx, v_perm, kq);
    ggml_tensor *kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    cur = ggml_cont_2d(ctx, kqv_merged, config.n_embd, n_tokens);

    return ggml_mul_mat(ctx, weights.output, cur);
}

} // namespace tllm::ops
