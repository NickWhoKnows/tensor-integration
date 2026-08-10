#include "tllm/ops/rope.h"

#include <cstring>

namespace tllm::ops
{

ggml_tensor *make_positions(ggml_context *ctx, const int n_tokens, const int offset)
{
    ggml_tensor *positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    int32_t *data = static_cast<int32_t *>(positions->data);
    for (int i = 0; i < n_tokens; ++i)
    {
        data[i] = offset + i;
    }
    return positions;
}

ggml_tensor *apply_rope(ggml_context *ctx, ggml_tensor *qkv, ggml_tensor *positions,
                        const model::Config &config)
{
    return ggml_rope_ext(ctx, qkv, positions, config.rope_freqs, config.rope_dimension_count,
                         GGML_ROPE_TYPE_NORMAL, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f,
                         1.0f, 0.0f, 0.0f);
}

} // namespace tllm::ops
