#include "tllm/gguf/loader.h"
#include "tllm/model/config.h"
#include "tllm/model/model.h"
#include "tllm/tokenizer/tokenizer.h"

#include "ggml.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void print_usage(const char *program) {
  std::cerr << "Usage: " << program << " [--model PATH] [--prompt TEXT] [--n N]\n"
            << "\n"
            << "  --model PATH    GGUF model file (default: "
               "models/Llama-3.2-1B-Instruct.gguf)\n"
            << "  --prompt TEXT   Prompt to tokenize and run through the model\n"
            << "  --n N           Number of tokens to generate (default: 10)\n";
}

struct Options {
  std::string model_path = "models/Llama-3.2-1B-Instruct.gguf";
  std::string prompt = "Hello";
  int n_generate = 10;
};

std::string resolve_model_path(const std::string &path) {
  if (std::ifstream(path).good()) {
    return path;
  }

  const std::string from_build = "../" + path;
  if (std::ifstream(from_build).good()) {
    return from_build;
  }

  return path;
}

Options parse_args(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      options.model_path = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      options.prompt = argv[++i];
    } else if ((arg == "--n" || arg == "-n") && i + 1 < argc) {
      options.n_generate = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  options.model_path = resolve_model_path(options.model_path);
  return options;
}

void run_inference(const Options &options) {
  tllm::gguf::Loader loader(options.model_path);
  const tllm::model::Config config = tllm::model::Config::from_gguf(loader);
  const tllm::tokenizer::Tokenizer tokenizer =
      tllm::tokenizer::Tokenizer::from_gguf(loader);

  std::cout << "Model: " << config.architecture << "\n"
            << "  layers: " << config.n_layer << "\n"
            << "  embed:  " << config.n_embd << "\n"
            << "  vocab:  " << tokenizer.vocab_size() << "\n"
            << "  params: " << loader.total_parameters() << "\n";

  tllm::model::Model model(loader, config, tokenizer);
  std::vector<int32_t> tokens = model.tokenize(options.prompt);

  std::cout << "Prompt: \"" << options.prompt << "\"\n"
            << "Input tokens (" << tokens.size() << "): ";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << tokens[i];
  }
  std::cout << "\n";

  ggml_context *ctx = loader.context();
  ggml_set_no_alloc(ctx, false);

  std::cout << "Generating " << options.n_generate << " tokens:\n";
  tokens = model.generate(ctx, std::move(tokens), options.n_generate);

  std::cout << "Generated text: \"" << tokenizer.decode(tokens) << "\"\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_args(argc, argv);
    run_inference(options);
  } catch (const std::exception &ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
