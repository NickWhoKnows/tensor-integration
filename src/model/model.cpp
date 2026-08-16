#include "tllm/model/model.h"

#include "tllm/model/weights.h"
#include "tllm/ops/embed.h"
#include "tllm/ops/norm.h"
#include "tllm/ops/rope.h"

#include "ggml-alloc.h"
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

void set_input(ggml_cgraph *graph, const char *name, const void *data, const size_t nbytes)
{
    ggml_tensor *tensor = ggml_graph_get_tensor(graph, name);
    if (tensor != nullptr && tensor->data != nullptr)
    {
        std::memcpy(tensor->data, data, nbytes);
    }
}

void set_inputs(ggml_cgraph *graph, const std::vector<int32_t> &tokens, const int position_offset)
{
    set_input(graph, "input_tokens", tokens.data(), tokens.size() * sizeof(int32_t));

    std::vector<int32_t> positions(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        positions[i] = position_offset + static_cast<int>(i);
    }
    set_input(graph, "rope_positions", positions.data(), positions.size() * sizeof(int32_t));
}

} // namespace

Model::Model(gguf::Loader &loader, const Config &config, const tokenizer::Tokenizer &tokenizer)
    : config_(config), tokenizer_(tokenizer)
{
    backend_ = ggml_backend_cpu_init();
    if (backend_ == nullptr)
    {
        throw std::runtime_error("failed to initialize CPU backend");
    }

    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (galloc_ == nullptr)
    {
        throw std::runtime_error("failed to initialize compute allocator");
    }

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

Model::~Model()
{
    if (galloc_ != nullptr)
    {
        ggml_gallocr_free(galloc_);
    }
    if (backend_ != nullptr)
    {
        ggml_backend_free(backend_);
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

    x = ops::rms_norm(ctx, x, output_norm_, config_.rms_norm_eps);
    return ggml_mul_mat(ctx, token_embd_, x);
}

void Model::run(ggml_context *ctx, ggml_tensor *root, const std::vector<int32_t> &tokens,
                const int position_offset, const runtime::KvCache *cache) const
{
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, root);

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

    if (!ggml_gallocr_alloc_graph(galloc_, graph))
    {
        throw std::runtime_error("failed to allocate compute graph");
    }

    set_inputs(graph, tokens, position_offset);
    ggml_backend_cpu_set_n_threads(backend_, 1);
    if (ggml_backend_graph_compute(backend_, graph) != GGML_STATUS_SUCCESS)
    {
        throw std::runtime_error("graph compute failed");
    }
}

std::vector<float> Model::logits_at(ggml_tensor *logits, const int position) const
{
    const int vocab = static_cast<int>(logits->ne[0]);
    std::vector<float> row(static_cast<size_t>(vocab));
    const size_t offset = static_cast<size_t>(position) * static_cast<size_t>(vocab) * sizeof(float);
    ggml_backend_tensor_get(logits, row.data(), offset, row.size() * sizeof(float));
    return row;
}

int32_t Model::predict_next(ggml_context *ctx, const std::vector<int32_t> &tokens,
                            runtime::KvCache *cache) const
{
    const int n_past = cache != nullptr ? cache->n_past() : 0;
    ggml_tensor *logits = forward(ctx, tokens, cache);
    ggml_tensor *predicted = ggml_argmax(ctx, logits);

    run(ctx, predicted, tokens, n_past, cache);

    int32_t next = 0;
    const size_t offset = static_cast<size_t>(predicted->ne[0] - 1) * sizeof(int32_t);
    ggml_backend_tensor_get(predicted, &next, offset, sizeof(int32_t));

    if (cache != nullptr)
    {
        cache->advance(static_cast<int>(tokens.size()));
    }
    return next;
}

void Model::print_top_logits(ggml_context *ctx, const std::vector<int32_t> &tokens, const int k) const
{
    ggml_tensor *logits = forward(ctx, tokens, nullptr);
    run(ctx, logits, tokens, 0, nullptr);

    const int pos = static_cast<int>(logits->ne[1]) - 1;
    const std::vector<float> row = logits_at(logits, pos);

    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(row.size());
    for (int id = 0; id < static_cast<int>(row.size()); ++id)
    {
        scored.emplace_back(row[static_cast<size_t>(id)], id);
    }

    const int top_k = std::min(k, static_cast<int>(scored.size()));
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
    cache.init(backend_, config_, std::min(config_.n_ctx, 2048));
    tokens.reserve(tokens.size() + static_cast<size_t>(n_new_tokens));

    std::vector<int32_t> step;
    for (int i = 0; i < n_new_tokens; ++i)
    {
        const std::vector<int32_t> &batch = i == 0 ? tokens : (step.assign(1, tokens.back()), step);
        tokens.push_back(predict_next(ctx, batch, &cache));
        if (tokens.back() == tokenizer_.eos_id())
        {
            break;
        }
    }

    return tokens;
}

} // namespace tllm::model
