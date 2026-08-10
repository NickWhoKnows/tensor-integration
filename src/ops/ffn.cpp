#include "tllm/ops/ffn.h"
#include "tllm/ops/norm.h"

namespace tllm::ops
{

ggml_tensor *ffn(ggml_context *ctx, ggml_tensor *x, const FfnWeights &weights, const float eps)
{
    ggml_tensor *cur = rms_norm(ctx, x, weights.norm, eps);
    ggml_tensor *gate_out = ggml_mul_mat(ctx, weights.gate, cur);
    ggml_tensor *up_out = ggml_mul_mat(ctx, weights.up, cur);
    ggml_tensor *hidden = ggml_mul(ctx, up_out, ggml_silu(ctx, gate_out));
    return ggml_mul_mat(ctx, weights.down, hidden);
}

} // namespace tllm::ops
