#include "tllm/cli.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace tllm::cli
{
namespace
{

void print_usage(const char *program)
{
    std::cerr << "Usage: " << program << " [--model PATH] [--prompt TEXT] [--system TEXT] [--n N] [--raw] [--verify]\n"
              << "  --model PATH    GGUF model file\n"
              << "  --prompt TEXT   User message\n"
              << "  --system TEXT   Optional system message\n"
              << "  --n N           Tokens to generate (default: 100)\n"
              << "  --raw           Skip Llama 3 chat template\n"
              << "  --verify        Print top-k logits only\n";
}

std::string resolve_model_path(const std::string &path)
{
    if (std::ifstream(path).good())
    {
        return path;
    }

    const std::string from_build = "../" + path;
    if (std::ifstream(from_build).good())
    {
        return from_build;
    }

    return path;
}

} // namespace

Options parse_args(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)
        {
            options.model_path = argv[++i];
        }
        else if (arg == "--prompt" && i + 1 < argc)
        {
            options.prompt = argv[++i];
        }
        else if (arg == "--system" && i + 1 < argc)
        {
            options.system_prompt = argv[++i];
        }
        else if (arg == "--raw")
        {
            options.use_chat_template = false;
        }
        else if ((arg == "--n" || arg == "-n") && i + 1 < argc)
        {
            options.n_generate = std::stoi(argv[++i]);
        }
        else if (arg == "--verify")
        {
            options.verify = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else
        {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    options.model_path = resolve_model_path(options.model_path);
    return options;
}

std::vector<int32_t> encode_prompt(const tokenizer::Tokenizer &tokenizer, const Options &options)
{
    if (!options.use_chat_template)
    {
        return tokenizer.encode(options.prompt);
    }

    std::vector<tokenizer::ChatMessage> messages;
    if (!options.system_prompt.empty())
    {
        messages.push_back({"system", options.system_prompt});
    }
    messages.push_back({"user", options.prompt});
    return tokenizer.encode_chat(messages);
}

} // namespace tllm::cli
