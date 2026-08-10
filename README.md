# tensor-integration

A minimal LLM inference engine built on [GGML](https://github.com/ggml-org/ggml). The project is being developed incrementally: each layer of the stack is added and validated before moving on to the next.

## Current status

| Component | Status |
|-----------|--------|
| GGUF loading (mmap, metadata, tensors) | Done |
| Model config from metadata | Done |
| BPE tokenizer from GGUF metadata | Done |
| Token embedding (`ggml_get_rows`) | Done |
| Full model forward (all layers) | Done |
| RMS norm | Done |
| Multi-head attention + GQA | Done |
| RoPE (NEOX + rope_freqs) | Done |
| SwiGLU FFN | Done |
| CPU graph execution | Done |
| RoPE | Done |
| Output head + logits | Done |
| Sampling / generation loop | Done (argmax) |
| KV cache | Planned |
| Metal backend | Planned |

## Project layout

```
tensor-integration/
├── include/tllm/
│   ├── gguf/loader.h       # mmap GGUF reader, zero-copy weight wrapping
│   ├── model/
│   │   ├── config.h        # hyperparameters from GGUF metadata
│   │   ├── weights.h       # tensor name helpers (blk.N.*)
│   │   └── model.h         # embed → N blocks forward pass
│   ├── ops/
│   │   ├── norm.h          # RMS norm
│   │   ├── attn.h          # multi-head attention
│   │   ├── ffn.h           # SwiGLU feed-forward
│   │   ├── embed.h         # token embedding lookup
│   │   └── trans.h         # transformer block (Layer)
│   ├── tokenizer/
│   │   └── tokenizer.h     # BPE tokenizer from GGUF metadata
│   └── runtime/
│       └── compute.h       # build + run ggml compute graphs
├── src/                    # implementations mirror include/tllm/
├── include/ggml/           # GGML submodule
├── models/                 # local GGUF weights (gitignored)
└── src/main.cpp            # CLI entry point
```

## Build

```bash
git submodule update --init --recursive
cmake -S . -B build
cmake --build build -j
```

## Run

Place a GGUF model in `models/` (e.g. Llama 3.2 1B Instruct), then:

```bash
./build/app
./build/app --model models/Llama-3.2-1B-Instruct.gguf --prompt "Hello"
```

Tokenizes the prompt, embeds tokens, and runs all transformer layers.

## Next steps

1. **RoPE** — apply `ggml_rope_ext` to Q/K before attention
2. **Output head** — `output_norm` + `output.weight` → logits
3. **Sampling** — argmax / temperature over logits
4. **Generation loop** — autoregressive token-by-token inference
5. **KV cache** — avoid recomputing past keys/values
