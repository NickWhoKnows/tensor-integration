#include "tllm/gguf/loader.h"
#include "tllm/model/config.h"
#include "tllm/model/model.h"
#include "tllm/runtime/backend.h"
#include "tllm/tokenizer/tokenizer.h"

#include "ggml.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void ggml_log_quiet(enum ggml_log_level level, const char *text, void *)
{
    if (level < GGML_LOG_LEVEL_WARN)
    {
        return;
    }

    fputs(text, stderr);
    fflush(stderr);
}

void print_usage(const char *program) {
  std::cerr << "Usage: " << program << " [--model PATH] [--prompt TEXT] [--n N] [--verify] [--cpu]\n"
            << "\n"
            << "  --model PATH    GGUF model file (default: "
               "models/Llama-3.2-1B-Instruct.gguf)\n"
            << "  --prompt TEXT   Prompt to tokenize and run through the model\n"
            << "  --n N           Number of tokens to generate (default: 10)\n"
            << "  --verify        Print config + top-k logits on CPU (no generation)\n"
            << "  --cpu           Force CPU backend for generation\n";
}

struct Options {
  std::string model_path = "models/Llama-3.2-1B-Instruct.gguf";
  std::string prompt = "Hello World";
  int n_generate = 640;
  bool verify = false;
  bool force_cpu = false;
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
    } else if (arg == "--verify") {
      options.verify = true;
    } else if (arg == "--cpu") {
      options.force_cpu = true;
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

  tllm::model::Model model(loader, config, tokenizer);

  std::cout << "Model: " << config.architecture << "\n"
            << "  layers: " << config.n_layer << "\n"
            << "  embed:  " << config.n_embd << "\n"
            << "  heads:  " << config.n_head << " (kv=" << config.n_head_kv << ")\n"
            << "  rope:   dim=" << config.rope_dimension_count
            << " base=" << config.rope_freq_base
            << " freqs=" << (model.config().rope_freqs ? "yes" : "no") << "\n"
            << "  vocab:  " << tokenizer.vocab_size() << "\n"
            << "  params: " << loader.total_parameters() << "\n";

  if (auto *embd = loader.find_tensor("token_embd.weight")) {
    std::cout << "  token_embd.type: " << embd->type << "\n";
  }

  std::vector<int32_t> tokens = tokenizer.encode(options.prompt);

  std::cout << "Prompt: \"" << options.prompt << "\"\n"
            << "Input tokens (" << tokens.size() << "): ";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << tokens[i] << "(\"" << tokenizer.token_to_piece(tokens[i]) << "\")";
  }
  std::cout << "\n";

  ggml_context *ctx = loader.context();

  if (options.verify) {
    tllm::runtime::Backend cpu_backend(tllm::runtime::Backend::Kind::Cpu);
    model.set_backend(&cpu_backend);
    model.print_top_logits(ctx, tokens, 10);
    std::cout << "\nReference (llama.cpp): top token id=0 \"!\" logit~18.96 for prompt \"Hello\"\n";
    return;
  }

  const auto backend_kind = options.force_cpu ? tllm::runtime::Backend::Kind::Cpu
                                              : tllm::runtime::Backend::Kind::Auto;
  tllm::runtime::Backend backend(backend_kind);
  loader.materialize_on_backend(backend.handle());
  model.set_backend(&backend);

  std::cout << "Backend: " << backend.name() << (backend.uses_gpu() ? " (GPU)" : " (CPU)") << "\n";

  tokens = model.generate(ctx, std::move(tokens), options.n_generate);
  loader.release_backend_weights();

  std::cout << tokenizer.format_generation(tokens) << "\n";
}

} // namespace

int main(int argc, char **argv) {
  ggml_log_set(ggml_log_quiet, nullptr);

  try {
    const Options options = parse_args(argc, argv);
    run_inference(options);
  } catch (const std::exception &ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
