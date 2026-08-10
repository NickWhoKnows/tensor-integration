#include "tllm/ops/norm.h"

namespace tllm::ops
{

ggml_tensor *rms_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *weight, const float eps)
{
    ggml_tensor *normalized = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, normalized, weight);
}

} // namespace tllm::ops
