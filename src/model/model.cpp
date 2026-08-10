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
#include <stdexcept>
#include <vector>

namespace tllm::model
{

namespace
{

void set_graph_input(ggml_cgraph *graph, const char *name, const void *data, const size_t nbytes,
                     const runtime::Backend *backend)
{
    ggml_tensor *tensor = ggml_graph_get_tensor(graph, name);
    if (tensor == nullptr)
    {
        return;
    }

    if (backend != nullptr)
    {
        backend->tensor_set(tensor, data, 0, nbytes);
    }
    else if (tensor->data != nullptr)
    {
        std::memcpy(tensor->data, data, nbytes);
    }
}

void set_graph_inputs(ggml_cgraph *graph, const std::vector<int32_t> &tokens, const int position_offset,
                      const runtime::Backend *backend)
{
    set_graph_input(graph, "input_tokens", tokens.data(), tokens.size() * sizeof(int32_t), backend);

    std::vector<int32_t> positions(static_cast<size_t>(tokens.size()));
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        positions[i] = position_offset + static_cast<int>(i);
    }
    set_graph_input(graph, "rope_positions", positions.data(), positions.size() * sizeof(int32_t),
                    backend);
}

void expand_kv_outputs(ggml_cgraph *graph, const runtime::KvCache *cache, const int n_layer)
{
    if (cache == nullptr)
    {
        return;
    }

    for (int i = 0; i < n_layer; ++i)
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

int32_t read_last_i32(ggml_tensor *tensor, const runtime::Backend *backend)
{
    const size_t offset = static_cast<size_t>(tensor->ne[0] - 1) * sizeof(int32_t);
    if (backend != nullptr)
    {
        int32_t value = 0;
        backend->tensor_get(tensor, &value, offset, sizeof(int32_t));
        return value;
    }

    return static_cast<const int32_t *>(tensor->data)[tensor->ne[0] - 1];
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

ggml_tensor *Model::forward(ggml_context *ctx, const std::vector<int32_t> &tokens,
                            runtime::KvCache *cache) const
{
    const int n_past = cache != nullptr ? cache->n_past() : 0;
    ggml_tensor *positions = ops::make_positions_input(ctx, static_cast<int>(tokens.size()));
    ggml_tensor *x = ops::embed_tokens(ctx, token_embd_, tokens);
    for (size_t i = 0; i < layers_.size(); ++i)
    {
        runtime::LayerKvCache *layer_cache = cache != nullptr ? &cache->layer(static_cast<int>(i)) : nullptr;
        x = layers_[i]->block_transformer(ctx, x, positions, layer_cache, n_past);
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
            set_graph_inputs(graph, *inputs.tokens, inputs.position_offset, backend_);
        }
        backend_->compute(graph);
        backend_->synchronize();
        return;
    }

    if (inputs.tokens != nullptr)
    {
        set_graph_inputs(graph, *inputs.tokens, inputs.position_offset, nullptr);
    }

    ggml_graph_compute_with_ctx(ctx, graph, 1);
}

void Model::compute(ggml_context *ctx, ggml_tensor *root, const std::vector<int32_t> &tokens,
                    const int position_offset, const runtime::KvCache *cache) const
{
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, root);
    expand_kv_outputs(graph, cache, config_.n_layer);
    execute_graph(ctx, graph, GraphInputs{.tokens = &tokens, .position_offset = position_offset});
}

int32_t Model::argmax_last(ggml_context *ctx, ggml_tensor *logits,
                           const std::vector<int32_t> &tokens, const int position_offset,
                           const runtime::KvCache *cache) const
{
    ggml_tensor *predicted = ggml_argmax(ctx, logits);
    compute(ctx, predicted, tokens, position_offset, cache);
    return read_last_i32(predicted, backend_);
}

int32_t Model::predict_next(ggml_context *ctx, const std::vector<int32_t> &tokens,
                            runtime::KvCache *cache) const
{
    const int n_past = cache != nullptr ? cache->n_past() : 0;
    const int32_t next = argmax_last(ctx, forward(ctx, tokens, cache), tokens, n_past, cache);
    if (cache != nullptr)
    {
        cache->commit(backend_, static_cast<int>(tokens.size()));
    }
    return next;
}

void Model::print_top_logits(ggml_context *ctx, const std::vector<int32_t> &tokens, const int k) const
{
    ggml_tensor *logits = forward(ctx, tokens);
    compute(ctx, logits, tokens, 0, nullptr);

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
    cache.init(*backend_, config_, std::min(config_.n_ctx, 2048));
    cache.reset();
    tokens.reserve(tokens.size() + static_cast<size_t>(n_new_tokens));

    std::vector<int32_t> step;
    for (int i = 0; i < n_new_tokens; ++i)
    {
        const std::vector<int32_t> &batch = i == 0 ? tokens : (step.assign(1, tokens.back()), step);
        const int32_t next = predict_next(ctx, batch, &cache);
        tokens.push_back(next);
        if (next == tokenizer_.eos_id())
        {
            break;
        }
    }

    return tokens;
}

} // namespace tllm::model
