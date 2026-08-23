#include "tllm/cli.h"
#include "tllm/gguf/loader.h"
#include "tllm/model/config.h"
#include "tllm/model/model.h"

#include "ggml.h"

#include <cstdlib>
#include <iostream>

namespace
{

void ggml_log_quiet(enum ggml_log_level level, const char *text, void *)
{
    if (level < GGML_LOG_LEVEL_WARN)
    {
        return;
    }

    fputs(text, stderr);
    fflush(stderr);
}

void print_tokens(const tllm::tokenizer::Tokenizer &tokenizer, const std::vector<int32_t> &tokens)
{
    std::cout << "Input tokens (" << tokens.size() << "): ";
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        if (i > 0)
        {
            std::cout << ", ";
        }
        std::cout << tokens[i] << "(\"" << tokenizer.token_to_piece(tokens[i]) << "\")";
    }
    std::cout << "\n";
}

} // namespace

int main(int argc, char **argv)
{
    ggml_log_set(ggml_log_quiet, nullptr);

    try
    {
        const tllm::cli::Options options = tllm::cli::parse_args(argc, argv);

        tllm::gguf::Loader loader(options.model_path);
        const tllm::model::Config config = tllm::model::Config::from_gguf(loader);
        const tllm::tokenizer::Tokenizer tokenizer = tllm::tokenizer::Tokenizer::from_gguf(loader);
        tllm::model::Model model(loader, config, tokenizer);

        std::cout << "Model: " << config.architecture << "\n"
                  << "  layers: " << config.n_layer << "\n"
                  << "  embed:  " << config.n_embd << "\n"
                  << "  heads:  " << config.n_head << " (kv=" << config.n_head_kv << ")\n"
                  << "  rope:   dim=" << config.rope_dimension_count << " base=" << config.rope_freq_base
                  << " freqs=" << (model.config().rope_freqs ? "yes" : "no") << "\n"
                  << "  vocab:  " << tokenizer.vocab_size() << "\n"
                  << "  params: " << loader.total_parameters() << "\n"
                  << "Prompt: \"" << options.prompt << "\"\n"
                  << "Chat template: " << (options.use_chat_template ? "on" : "off") << "\n";

        std::vector<int32_t> tokens = tllm::cli::encode_prompt(tokenizer, options);
        print_tokens(tokenizer, tokens);

        ggml_context *ctx = loader.context();
        const size_t prompt_token_count = tokens.size();
        tokens = model.generate(ctx, std::move(tokens), options.n_generate);
        std::cout << "\n\n" << tokenizer.format_generation(tokens, prompt_token_count) << "\n";
    }
    catch (const std::exception &ex)
    {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
