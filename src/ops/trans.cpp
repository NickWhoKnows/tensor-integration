#include "tllm/ops/trans.h"

#include "tllm/model/weights.h"

namespace tllm::ops
{

Layer::Layer(const int index, ggml_context * /*ctx*/, const model::Config &config,
             gguf::Loader &loader)
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
                                      runtime::LayerKvCache *cache,
                                      const int position_offset) const
{
    ggml_tensor *attn_out = attention(ctx, x, attn_, config_, positions, cache, position_offset);
    ggml_tensor *x_attn = ggml_add(ctx, x, attn_out);
    ggml_tensor *ffn_out = ffn(ctx, x_attn, ffn_, config_.rms_norm_eps);
    return ggml_add(ctx, x_attn, ffn_out);
}

} // namespace tllm::ops
