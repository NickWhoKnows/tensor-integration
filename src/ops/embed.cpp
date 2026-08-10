#include "tllm/ops/embed.h"

#include <cstring>

namespace tllm::ops
{

ggml_tensor *embed_tokens(ggml_context *ctx, ggml_tensor *token_embd, const std::vector<int32_t> &tokens)
{
    ggml_tensor *ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(tokens.size()));
    std::memcpy(ids->data, tokens.data(), tokens.size() * sizeof(int32_t));
    return ggml_get_rows(ctx, token_embd, ids);
}

} // namespace tllm::ops
