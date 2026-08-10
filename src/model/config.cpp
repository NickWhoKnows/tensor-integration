#include "tllm/model/config.h"

#include <stdexcept>

namespace tllm::model
{
namespace
{

template <typename T>
const T *get_scalar(const gguf::MetaValue *value)
{
    return value != nullptr ? std::get_if<T>(value) : nullptr;
}

int read_int(const gguf::Loader &loader, const std::string &key)
{
    const gguf::MetaValue *value = loader.find_metadata(key);
    if (value == nullptr)
    {
        throw std::runtime_error("missing model metadata key: " + key);
    }

    if (const auto *v = get_scalar<uint32_t>(value))
    {
        return static_cast<int>(*v);
    }
    if (const auto *v = get_scalar<int32_t>(value))
    {
        return *v;
    }
    if (const auto *v = get_scalar<uint64_t>(value))
    {
        return static_cast<int>(*v);
    }
    if (const auto *v = get_scalar<int64_t>(value))
    {
        return static_cast<int>(*v);
    }

    throw std::runtime_error("metadata key has unsupported integer type: " + key);
}

float read_float(const gguf::Loader &loader, const std::string &key, const float fallback)
{
    const gguf::MetaValue *value = loader.find_metadata(key);
    if (value == nullptr)
    {
        return fallback;
    }

    if (const auto *v = get_scalar<float>(value))
    {
        return *v;
    }
    if (const auto *v = get_scalar<double>(value))
    {
        return static_cast<float>(*v);
    }

    throw std::runtime_error("metadata key has unsupported float type: " + key);
}

std::string read_string(const gguf::Loader &loader, const std::string &key)
{
    const gguf::MetaValue *value = loader.find_metadata(key);
    if (value == nullptr)
    {
        throw std::runtime_error("missing model metadata key: " + key);
    }

    if (const auto *v = get_scalar<std::string>(value))
    {
        return *v;
    }

    throw std::runtime_error("metadata key has unsupported string type: " + key);
}

} // namespace

Config Config::from_gguf(const gguf::Loader &loader)
{
    Config config;
    config.architecture = read_string(loader, "general.architecture");
    config.n_embd = read_int(loader, "llama.embedding_length");
    config.n_ff = read_int(loader, "llama.feed_forward_length");
    config.n_head = read_int(loader, "llama.attention.head_count");
    config.n_head_kv = read_int(loader, "llama.attention.head_count_kv");
    config.n_layer = read_int(loader, "llama.block_count");
    config.n_ctx = read_int(loader, "llama.context_length");
    config.rms_norm_eps = read_float(loader, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    config.rope_freq_base = read_float(loader, "llama.rope.freq_base", 10000.0f);
    config.rope_dimension_count = read_int(loader, "llama.rope.dimension_count");
    return config;
}

} // namespace tllm::model
