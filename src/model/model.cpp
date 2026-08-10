#include "tllm/model/model.h"

#include "tllm/model/weights.h"
#include "tllm/ops/embed.h"
#include "tllm/ops/norm.h"
#include "tllm/ops/rope.h"
#include "tllm/runtime/backend.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

namespace tllm::model
{

namespace
{

constexpr float kRepetitionPenalty = 1.15f;
constexpr int kRepetitionWindow = 64;

void apply_repetition_penalty(std::vector<float> &row, const std::vector<int32_t> &history)
{
    const int start = std::max(0, static_cast<int>(history.size()) - kRepetitionWindow);
    for (size_t i = static_cast<size_t>(start); i < history.size(); ++i)
    {
        const int id = history[static_cast<size_t>(i)];
        if (id < 0 || id >= static_cast<int>(row.size()))
        {
            continue;
        }

        float &logit = row[static_cast<size_t>(id)];
        if (logit > 0.0f)
        {
            logit /= kRepetitionPenalty;
        }
        else
        {
            logit *= kRepetitionPenalty;
        }
    }
}

} // namespace

namespace
{

int practical_ctx(const Config &config)
{
    return std::min(config.n_ctx, 2048);
}

void set_input_tokens(ggml_cgraph *graph, const std::vector<int32_t> &tokens, const runtime::Backend *backend)
{
    ggml_tensor *input = ggml_graph_get_tensor(graph, "input_tokens");
    if (input == nullptr)
    {
        return;
    }

    const size_t nbytes = tokens.size() * sizeof(int32_t);
    if (backend != nullptr)
    {
        backend->tensor_set(input, tokens.data(), 0, nbytes);
    }
    else if (input->data != nullptr)
    {
        std::memcpy(input->data, tokens.data(), nbytes);
    }
}

void set_rope_positions(ggml_cgraph *graph, const int n_tokens, const int position_offset,
                        const runtime::Backend *backend)
{
    ggml_tensor *positions = ggml_graph_get_tensor(graph, "rope_positions");
    if (positions == nullptr)
    {
        return;
    }

    std::vector<int32_t> data(static_cast<size_t>(n_tokens));
    for (int i = 0; i < n_tokens; ++i)
    {
        data[static_cast<size_t>(i)] = position_offset + i;
    }

    const size_t nbytes = data.size() * sizeof(int32_t);
    if (backend != nullptr)
    {
        backend->tensor_set(positions, data.data(), 0, nbytes);
    }
    else if (positions->data != nullptr)
    {
        std::memcpy(positions->data, data.data(), nbytes);
    }
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
    ggml_tensor *positions = ops::make_positions_input(ctx, static_cast<int>(tokens.size()));
    ggml_tensor *x = ops::embed_tokens(ctx, token_embd_, tokens);
    for (const auto &layer : layers_)
    {
        x = layer->block_transformer(ctx, x, positions);
    }

    return final_output(ctx, x);
}

ggml_tensor *Model::forward(ggml_context *ctx, const std::vector<int32_t> &tokens,
                            runtime::KvCache &cache) const
{
    const int n_past = cache.n_past();
    ggml_tensor *positions = ops::make_positions_input(ctx, static_cast<int>(tokens.size()));
    ggml_tensor *x = ops::embed_tokens(ctx, token_embd_, tokens);
    for (size_t i = 0; i < layers_.size(); ++i)
    {
        x = layers_[i]->block_transformer(ctx, x, positions, &cache.layer(static_cast<int>(i)), n_past);
    }

    return final_output(ctx, x);
}

ggml_tensor *Model::final_output(ggml_context *ctx, ggml_tensor *x) const
{
    x = ops::rms_norm(ctx, x, output_norm_, config_.rms_norm_eps);
    return ggml_mul_mat(ctx, token_embd_, x);
}

void Model::execute_graph(ggml_context *ctx, ggml_cgraph *graph,
                          const GraphInputs &inputs) const
{
    if (backend_ != nullptr)
    {
        backend_->alloc_graph(graph);
        if (inputs.tokens != nullptr)
        {
            set_input_tokens(graph, *inputs.tokens, backend_);
            set_rope_positions(graph, static_cast<int>(inputs.tokens->size()), inputs.position_offset,
                               backend_);
        }
        backend_->compute(graph);
        backend_->synchronize();
        return;
    }

    if (inputs.tokens != nullptr)
    {
        set_input_tokens(graph, *inputs.tokens, nullptr);
        set_rope_positions(graph, static_cast<int>(inputs.tokens->size()), inputs.position_offset,
                           nullptr);
    }

    ggml_graph_compute_with_ctx(ctx, graph, 1);
}

int32_t Model::argmax_last(ggml_context *ctx, ggml_tensor *logits,
                           const std::vector<int32_t> &tokens, const int position_offset,
                           const runtime::KvCache *cache) const
{
    ggml_tensor *predicted = ggml_argmax(ctx, logits);

    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, predicted);
    if (cache != nullptr)
    {
        for (int i = 0; i < config_.n_layer; ++i)
        {
            const runtime::LayerKvCache &layer = cache->layer(i);
            if (layer.last_k != nullptr)
            {
                ggml_build_forward_expand(graph, layer.last_k);
            }
            if (layer.last_v != nullptr)
            {
                ggml_build_forward_expand(graph, layer.last_v);
            }
        }
    }
    execute_graph(ctx, graph, GraphInputs{.tokens = &tokens, .position_offset = position_offset});

    int32_t predicted_id = 0;
    if (backend_ != nullptr)
    {
        backend_->tensor_get(predicted, &predicted_id,
                             static_cast<size_t>(predicted->ne[0] - 1) * sizeof(int32_t),
                             sizeof(int32_t));
    }
    else
    {
        const int32_t *predicted_ids = static_cast<const int32_t *>(predicted->data);
        predicted_id = predicted_ids[predicted->ne[0] - 1];
    }

    return predicted_id;
}

int32_t Model::argmax_with_penalty(ggml_context *ctx, ggml_tensor *logits,
                                   const std::vector<int32_t> &tokens, const int position_offset,
                                   const runtime::KvCache *cache,
                                   const std::vector<int32_t> &history) const
{
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, logits);
    if (cache != nullptr)
    {
        for (int i = 0; i < config_.n_layer; ++i)
        {
            const runtime::LayerKvCache &layer = cache->layer(i);
            if (layer.last_k != nullptr)
            {
                ggml_build_forward_expand(graph, layer.last_k);
            }
            if (layer.last_v != nullptr)
            {
                ggml_build_forward_expand(graph, layer.last_v);
            }
        }
    }
    execute_graph(ctx, graph, GraphInputs{.tokens = &tokens, .position_offset = position_offset});

    const int vocab = static_cast<int>(logits->ne[0]);
    const int pos = static_cast<int>(logits->ne[1]) - 1;

    std::vector<float> row(static_cast<size_t>(vocab));
    const size_t row_offset = static_cast<size_t>(pos) * static_cast<size_t>(vocab) * sizeof(float);
    if (backend_ != nullptr)
    {
        backend_->tensor_get(logits, row.data(), row_offset, row.size() * sizeof(float));
    }
    else
    {
        const float *src = static_cast<const float *>(logits->data) + static_cast<size_t>(pos) * vocab;
        std::copy(src, src + vocab, row.begin());
    }

    apply_repetition_penalty(row, history);

    return static_cast<int32_t>(
        std::distance(row.begin(), std::max_element(row.begin(), row.end())));
}

void Model::commit_cache(runtime::KvCache &cache, const int n_tokens) const
{
    if (cache.n_past() + n_tokens > cache.max_ctx())
    {
        throw std::runtime_error("kv cache overflow");
    }

    const int head_dim = config_.n_embd / config_.n_head;
    for (int i = 0; i < config_.n_layer; ++i)
    {
        cache.commit_layer(i, head_dim, config_.n_head_kv, backend_);
    }
    cache.advance(n_tokens);
}

int32_t Model::predict_next(ggml_context *ctx, const std::vector<int32_t> &tokens) const
{
    ggml_tensor *logits = forward(ctx, tokens);
    return argmax_last(ctx, logits, tokens, 0);
}

int32_t Model::predict_next(ggml_context *ctx, const std::vector<int32_t> &tokens,
                            runtime::KvCache &cache, const std::vector<int32_t> &history) const
{
    const int n_past = cache.n_past();
    ggml_tensor *logits = forward(ctx, tokens, cache);
    const int32_t next = argmax_with_penalty(ctx, logits, tokens, n_past, &cache, history);
    commit_cache(cache, static_cast<int>(tokens.size()));
    return next;
}

void Model::print_top_logits(ggml_context *ctx, const std::vector<int32_t> &tokens, const int k) const
{
    ggml_tensor *logits = forward(ctx, tokens);

    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, logits);
    execute_graph(ctx, graph, GraphInputs{.tokens = &tokens, .position_offset = 0});

    const int vocab = static_cast<int>(logits->ne[0]);
    const int pos = static_cast<int>(logits->ne[1]) - 1;

    std::vector<float> row(static_cast<size_t>(vocab));
    const size_t row_offset = static_cast<size_t>(pos) * static_cast<size_t>(vocab) * sizeof(float);
    if (backend_ != nullptr)
    {
        backend_->tensor_get(logits, row.data(), row_offset, row.size() * sizeof(float));
    }
    else
    {
        const float *src = static_cast<const float *>(logits->data) + static_cast<size_t>(pos) * vocab;
        std::copy(src, src + vocab, row.begin());
    }

    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(static_cast<size_t>(vocab));
    for (int id = 0; id < vocab; ++id)
    {
        scored.emplace_back(row[static_cast<size_t>(id)], id);
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
    if (backend_ == nullptr)
    {
        throw std::runtime_error("metal/cpu backend required for generation");
    }

    runtime::KvCache cache;
    cache.init(*backend_, config_, practical_ctx(config_));
    cache.reset();

    tokens.reserve(tokens.size() + static_cast<size_t>(n_new_tokens));

    int32_t next = predict_next(ctx, tokens, cache, tokens);
    tokens.push_back(next);
    if (next == tokenizer_.eos_id())
    {
        return tokens;
    }

    for (int i = 1; i < n_new_tokens; ++i)
    {
        next = predict_next(ctx, {next}, cache, tokens);
        tokens.push_back(next);

        if (next == tokenizer_.eos_id())
        {
            break;
        }
    }

    return tokens;
}

} // namespace tllm::model
