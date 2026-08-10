#include "tllm/ops/trans.h"

#include <iostream>

namespace tllm::ops {

Layer::~Layer() {}

ggml_tensor *Layer::block_transformer(ggml_context *ctx, ggml_tensor *x) {
  // attn wiht rms norm
  ggml_tensor *attn_out = attention(ctx, x, attn_weights_, config_);
  // first residual
  ggml_tensor *after_attn = ggml_add(ctx, x, attn_out);
  // ffn with rms norm
  ggml_tensor *ffn_out =
      ffn(ctx, after_attn, ffn_weights_, config_.rms_norm_eps);
  // second residual
  ggml_tensor *after_ffn = ggml_add(ctx, after_attn, ffn_out);
  return after_ffn;
}

void Layer::print_attn_weights() {

  const float *attn_values =
      static_cast<const float *>(attn_weights_.output->data);
  std::cout << "Attention block " << block_index_ << " output ["
            << attn_weights_.output->ne[0] << ", "
            << attn_weights_.output->ne[1] << "]: " << attn_values[0] << ", "
            << attn_values[1] << "\n";
}

void Layer::print_ffn_weights() {
  std::cout << "FFN Weights: " << ffn_weights_.norm->name << std::endl;
  const float *ffn_values = static_cast<const float *>(ffn_weights_.down->data);
  std::cout << "FFN block " << block_index_ << " output ["
            << ffn_weights_.down->ne[0] << ", " << ffn_weights_.down->ne[1]
            << "]: " << ffn_values[0] << ", " << ffn_values[1] << "\n";
}

} // namespace tllm::ops