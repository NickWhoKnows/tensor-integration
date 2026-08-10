#include "tllm/ops/embed.h"

namespace tllm::ops
{

ggml_tensor *embed_tokens(ggml_context *ctx, ggml_tensor *token_embd, const std::vector<int32_t> &tokens)
{
    ggml_tensor *ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(tokens.size()));
    ggml_set_name(ids, "input_tokens");
    ggml_set_input(ids);
    return ggml_get_rows(ctx, token_embd, ids);
}

} // namespace tllm::ops
