#include "tllm/model/model.h"

#include "tllm/model/weights.h"
#include "tllm/ops/embed.h"
#include "tllm/ops/norm.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <iostream>
#include <utility>

namespace tllm::model
{

namespace
{

int practical_ctx(const Config &config)
{
    return std::min(config.n_ctx, 2048);
}

} // namespace

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

    return final_output(ctx, x);
}

ggml_tensor *Model::forward(ggml_context *ctx, const std::vector<int32_t> &tokens,
                            runtime::KvCache &cache) const
{
    const int n_past = cache.n_past();
    ggml_tensor *x = ops::embed_tokens(ctx, token_embd_, tokens);
    for (size_t i = 0; i < layers_.size(); ++i)
    {
        x = layers_[i]->block_transformer(ctx, x, &cache.layer(static_cast<int>(i)), n_past);
    }

    return final_output(ctx, x);
}

ggml_tensor *Model::final_output(ggml_context *ctx, ggml_tensor *x) const
{
    x = ops::rms_norm(ctx, x, output_norm_, config_.rms_norm_eps);
    return ggml_mul_mat(ctx, token_embd_, x);
}

int32_t Model::argmax_last(ggml_context *ctx, ggml_tensor *logits) const
{
    ggml_tensor *predicted = ggml_argmax(ctx, logits);

    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, predicted);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    const int32_t *predicted_ids = static_cast<const int32_t *>(predicted->data);
    return predicted_ids[predicted->ne[0] - 1];
}

void Model::commit_cache(runtime::KvCache &cache, const int n_tokens) const
{
    const int head_dim = config_.n_embd / config_.n_head;
    for (int i = 0; i < config_.n_layer; ++i)
    {
        cache.commit_layer(i, head_dim, config_.n_head_kv);
    }
    cache.advance(n_tokens);
}

int32_t Model::predict_next(ggml_context *ctx, const std::vector<int32_t> &tokens) const
{
    ggml_tensor *logits = forward(ctx, tokens);
    return argmax_last(ctx, logits);
}

int32_t Model::predict_next(ggml_context *ctx, const std::vector<int32_t> &tokens,
                            runtime::KvCache &cache) const
{
    ggml_tensor *logits = forward(ctx, tokens, cache);
    ggml_tensor *predicted = ggml_argmax(ctx, logits);

    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, predicted);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    commit_cache(cache, static_cast<int>(tokens.size()));

    const int32_t *predicted_ids = static_cast<const int32_t *>(predicted->data);
    return predicted_ids[predicted->ne[0] - 1];
}

void Model::print_top_logits(ggml_context *ctx, const std::vector<int32_t> &tokens, const int k) const
{
    ggml_tensor *logits = forward(ctx, tokens);

    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, logits);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    const int vocab = static_cast<int>(logits->ne[0]);
    const int pos = static_cast<int>(logits->ne[1]) - 1;
    const float *row = static_cast<const float *>(logits->data) + static_cast<size_t>(pos) * vocab;

    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(static_cast<size_t>(vocab));
    for (int id = 0; id < vocab; ++id)
    {
        scored.emplace_back(row[id], id);
    }

    const int top_k = std::min(k, vocab);
    std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });

    std::cout << "Top-" << top_k << " logits at position " << pos << ":\n";
    for (int i = 0; i < top_k; ++i)
    {
        const int32_t id = scored[static_cast<size_t>(i)].second;
        std::cout << "  " << (i + 1) << ": id=" << id << " logit=" << scored[static_cast<size_t>(i)].first
                  << " piece=\"" << tokenizer_.token_to_piece(id) << "\"\n";
    }
}

std::vector<int32_t> Model::generate(ggml_context *ctx, std::vector<int32_t> tokens,
                                     const int n_new_tokens) const
{
    runtime::KvCache cache;
    cache.init(ctx, config_, practical_ctx(config_));
    cache.reset();

    tokens.reserve(tokens.size() + static_cast<size_t>(n_new_tokens));

    int32_t next = predict_next(ctx, tokens, cache);
    tokens.push_back(next);
    std::cout << "  step 1: id=" << next << " piece=\"" << tokenizer_.token_to_piece(next) << "\"\n";
    if (next == tokenizer_.eos_id())
    {
        return tokens;
    }

    for (int i = 1; i < n_new_tokens; ++i)
    {
        next = predict_next(ctx, {next}, cache);
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
