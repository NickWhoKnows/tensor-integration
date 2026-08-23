#include "tllm/model/config.h"

#include "tllm/gguf/metadata.h"

namespace tllm::model
{

Config Config::from_gguf(const gguf::Loader &loader)
{
    Config config;
    config.architecture = gguf::meta::require_string(loader, "general.architecture");
    config.n_embd = gguf::meta::require_int(loader, "llama.embedding_length");
    config.n_head = gguf::meta::require_int(loader, "llama.attention.head_count");
    config.n_head_kv = gguf::meta::require_int(loader, "llama.attention.head_count_kv");
    config.n_layer = gguf::meta::require_int(loader, "llama.block_count");
    config.n_ctx = gguf::meta::require_int(loader, "llama.context_length");
    config.rope_dimension_count = gguf::meta::require_int(loader, "llama.rope.dimension_count");
    config.rms_norm_eps = gguf::meta::read_float_or(loader, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    config.rope_freq_base = gguf::meta::read_float_or(loader, "llama.rope.freq_base", 10000.0f);
    return config;
}

} // namespace tllm::model
