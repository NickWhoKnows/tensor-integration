#include "tllm/model/model.h"

#include "tllm/model/weights.h"
#include "tllm/ops/embed.h"
#include "tllm/ops/norm.h"

#include "ggml-cpu.h"

#include <iostream>

namespace tllm::model
{

Model::Model(gguf::Loader &loader, const Config &config, const tokenizer::Tokenizer &tokenizer)
    : config_(config), tokenizer_(tokenizer)
{
    if (loader.find_tensor("rope_freqs.weight") != nullptr)
    {
        config_.rope_freqs = loader.wrap_tensor("rope_freqs.weight");
    }

    token_embd_ = loader.wrap_tensor(token_embedding());
    output_norm_ = loader.wrap_tensor(output_norm());

    layers_.reserve(static_cast<size_t>(config.n_layer));
    for (int i = 0; i < config.n_layer; ++i)
    {
        layers_.push_back(std::make_unique<ops::Layer>(i, loader.context(), config, loader));
    }
}

std::vector<int32_t> Model::tokenize(const std::string &prompt, const bool add_bos) const
{
    return tokenizer_.encode(prompt, add_bos, false);
}

ggml_tensor *Model::forward(ggml_context *ctx, const std::vector<int32_t> &tokens) const
{
    ggml_tensor *x = ops::embed_tokens(ctx, token_embd_, tokens);
    for (const auto &layer : layers_)
    {
        x = layer->block_transformer(ctx, x);
    }

    x = final_output(ctx, x);
    return x;
}

ggml_tensor *Model::final_output(ggml_context *ctx, ggml_tensor *x) const
{
    x = ops::rms_norm(ctx, x, output_norm_, config_.rms_norm_eps);
    // Llama ties the LM head to token_embd.weight (no separate output.weight tensor).
    return ggml_mul_mat(ctx, token_embd_, x);
}

int32_t Model::predict_next(ggml_context *ctx, const std::vector<int32_t> &tokens) const
{
    ggml_tensor *logits = forward(ctx, tokens);
    ggml_tensor *predicted = ggml_argmax(ctx, logits);

    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, predicted);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    const int32_t *predicted_ids = static_cast<const int32_t *>(predicted->data);
    return predicted_ids[predicted->ne[0] - 1];
}

std::vector<int32_t> Model::generate(ggml_context *ctx, std::vector<int32_t> tokens, const int n_new_tokens) const
{
    tokens.reserve(tokens.size() + static_cast<size_t>(n_new_tokens));

    for (int i = 0; i < n_new_tokens; ++i)
    {
        const int32_t next = predict_next(ctx, tokens);
        tokens.push_back(next);

        std::cout << "  step " << (i + 1) << ": id=" << next
                  << " piece=\"" << tokenizer_.token_to_piece(next) << "\"\n";

        if (next == tokenizer_.eos_id())
        {
            break;
        }
    }

    return tokens;
}

} // namespace tllm::model
