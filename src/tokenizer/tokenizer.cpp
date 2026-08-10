#include "tllm/tokenizer/tokenizer.h"

#include "tllm/gguf/metadata.h"

#include <algorithm>
#include <cctype>
#include <queue>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace tllm::tokenizer
{
namespace
{

struct Bigram
{
    int left = -1;
    int right = -1;
    int rank = 0;
    std::string text;
};

struct Symbol
{
    const char *text = nullptr;
    size_t n = 0;
    int prev = -1;
    int next = -1;
};

struct BigramCompare
{
    bool operator()(const Bigram &a, const Bigram &b) const
    {
        if (a.rank != b.rank)
        {
            return a.rank > b.rank;
        }
        return a.left > b.left;
    }
};

bool is_llama_bpe_pre(const std::string &pre)
{
    return pre == "llama-bpe" || pre == "llama3" || pre == "llama-v3";
}

std::string normalize_llama_piece(const std::string &piece)
{
    std::string out;
    out.reserve(piece.size());

    size_t i = 0;
    while (i < piece.size() && piece[i] == ' ')
    {
        out.push_back(static_cast<char>(0xC4));
        out.push_back(static_cast<char>(0xA0));
        ++i;
    }

    out.append(piece, i, std::string::npos);
    return out;
}

} // namespace

Tokenizer Tokenizer::from_gguf(const gguf::Loader &loader)
{
    Tokenizer tokenizer;
    tokenizer.id_to_token_ = gguf::meta::read_string_array(loader, "tokenizer.ggml.tokens");

    for (size_t i = 0; i < tokenizer.id_to_token_.size(); ++i)
    {
        tokenizer.token_to_id_.emplace(tokenizer.id_to_token_[i], static_cast<int32_t>(i));
    }

    const auto merges = gguf::meta::read_string_array(loader, "tokenizer.ggml.merges");
    tokenizer.load_bpe_ranks(merges);

    const auto pre = gguf::meta::read_string(loader, "tokenizer.ggml.pre").value_or("default");
    tokenizer.ignore_merges_ = is_llama_bpe_pre(pre);

    tokenizer.bos_id_ = gguf::meta::read_int(loader, "tokenizer.ggml.bos_token_id").value_or(-1);
    tokenizer.eos_id_ = gguf::meta::read_int(loader, "tokenizer.ggml.eos_token_id").value_or(-1);

    return tokenizer;
}

void Tokenizer::load_bpe_ranks(const std::vector<std::string> &merges)
{
    for (size_t i = 0; i < merges.size(); ++i)
    {
        const size_t space = merges[i].find(' ');
        if (space == std::string::npos)
        {
            continue;
        }

        const std::string left = merges[i].substr(0, space);
        const std::string right = merges[i].substr(space + 1);
        bpe_ranks_.emplace(left + " " + right, static_cast<int>(i));
    }
}

int Tokenizer::find_bpe_rank(const std::string &left, const std::string &right) const
{
    const auto it = bpe_ranks_.find(left + " " + right);
    return it == bpe_ranks_.end() ? -1 : it->second;
}

std::vector<std::string> Tokenizer::pretokenize(const std::string &text) const
{
    static const std::regex k_llama3_regex(
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n[:alpha:][:digit:]]?[[:alpha:]]+|\\d{1,3}| ?[^\\s[:alpha:][:digit:]]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        std::regex::ECMAScript | std::regex::optimize);

    std::vector<std::string> pieces;
    auto begin = std::sregex_iterator(text.begin(), text.end(), k_llama3_regex);
    const auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it)
    {
        if (!it->str().empty())
        {
            pieces.push_back(it->str());
        }
    }

    if (pieces.empty() && !text.empty())
    {
        pieces.push_back(text);
    }

    return pieces;
}

std::vector<int32_t> Tokenizer::encode_piece(const std::string &piece) const
{
    const std::string vocab_piece = ignore_merges_ ? normalize_llama_piece(piece) : piece;

    if (ignore_merges_)
    {
        const auto whole = token_to_id_.find(vocab_piece);
        if (whole != token_to_id_.end())
        {
            return {whole->second};
        }
    }

    std::vector<Symbol> symbols;
    symbols.reserve(vocab_piece.size());

    size_t offset = 0;
    int index = 0;
    while (offset < vocab_piece.size())
    {
        size_t char_len = 1;
        const unsigned char c = static_cast<unsigned char>(vocab_piece[offset]);
        if ((c & 0xE0) == 0xC0)
        {
            char_len = 2;
        }
        else if ((c & 0xF0) == 0xE0)
        {
            char_len = 3;
        }
        else if ((c & 0xF8) == 0xF0)
        {
            char_len = 4;
        }
        char_len = std::min(char_len, vocab_piece.size() - offset);

        Symbol sym;
        sym.text = vocab_piece.c_str() + offset;
        sym.n = char_len;
        sym.prev = index - 1;
        sym.next = (offset + char_len >= vocab_piece.size()) ? -1 : index + 1;
        symbols.push_back(sym);

        offset += char_len;
        ++index;
    }

    auto add_bigram = [&](const int left, const int right, std::priority_queue<Bigram, std::vector<Bigram>, BigramCompare> &queue) {
        if (left < 0 || right < 0 || left >= static_cast<int>(symbols.size()) || right >= static_cast<int>(symbols.size()))
        {
            return;
        }

        const auto &left_sym = symbols[left];
        const auto &right_sym = symbols[right];
        if (left_sym.n == 0 || right_sym.n == 0)
        {
            return;
        }

        const std::string left_token(left_sym.text, left_sym.n);
        const std::string right_token(right_sym.text, right_sym.n);
        const int rank = find_bpe_rank(left_token, right_token);
        if (rank < 0)
        {
            return;
        }

        Bigram bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.rank = rank;
        bigram.text = left_token + right_token;
        queue.push(bigram);
    };

    std::priority_queue<Bigram, std::vector<Bigram>, BigramCompare> queue;
    for (int i = 1; i < static_cast<int>(symbols.size()); ++i)
    {
        add_bigram(i - 1, i, queue);
    }

    while (!queue.empty())
    {
        const Bigram bigram = queue.top();
        queue.pop();

        auto &left_sym = symbols[bigram.left];
        auto &right_sym = symbols[bigram.right];
        if (left_sym.n == 0 || right_sym.n == 0)
        {
            continue;
        }

        const std::string left_token(left_sym.text, left_sym.n);
        const std::string right_token(right_sym.text, right_sym.n);
        if (left_token + right_token != bigram.text)
        {
            continue;
        }

        left_sym.n += right_sym.n;
        right_sym.n = 0;
        left_sym.next = right_sym.next;
        if (right_sym.next >= 0)
        {
            symbols[right_sym.next].prev = bigram.left;
        }

        add_bigram(left_sym.prev, bigram.left, queue);
        add_bigram(bigram.left, left_sym.next, queue);
    }

    std::vector<int32_t> tokens;
    for (const auto &sym : symbols)
    {
        if (sym.n == 0)
        {
            continue;
        }

        const std::string token_text(sym.text, sym.n);
        const auto it = token_to_id_.find(token_text);
        if (it == token_to_id_.end())
        {
            throw std::runtime_error("unknown tokenizer piece: " + token_text);
        }
        tokens.push_back(it->second);
    }

    return tokens;
}

std::vector<int32_t> Tokenizer::encode(const std::string &text, const bool add_bos, const bool add_eos) const
{
    std::vector<int32_t> tokens;
    if (add_bos && bos_id_ >= 0)
    {
        tokens.push_back(bos_id_);
    }

    for (const auto &piece : pretokenize(text))
    {
        const auto piece_tokens = encode_piece(piece);
        tokens.insert(tokens.end(), piece_tokens.begin(), piece_tokens.end());
    }

    if (add_eos && eos_id_ >= 0)
    {
        tokens.push_back(eos_id_);
    }

    return tokens;
}

std::string Tokenizer::token_to_piece(const int32_t token) const
{
    if (token < 0 || token >= static_cast<int32_t>(id_to_token_.size()))
    {
        return "";
    }
    return id_to_token_[static_cast<size_t>(token)];
}

std::string Tokenizer::piece_to_text(const std::string &piece) const
{
    std::string text;
    for (size_t i = 0; i < piece.size();)
    {
        if (i + 1 < piece.size() && static_cast<unsigned char>(piece[i]) == 0xC4)
        {
            const unsigned char next = static_cast<unsigned char>(piece[i + 1]);
            if (next == 0xA0)
            {
                text += ' ';
                i += 2;
                continue;
            }
            if (next == 0x8A)
            {
                text += '\n';
                i += 2;
                continue;
            }
        }

        text += piece[i++];
    }

    return text;
}

std::string Tokenizer::decode(const std::vector<int32_t> &tokens) const
{
    std::string text;
    for (const int32_t token : tokens)
    {
        if (token == bos_id_ || token == eos_id_)
        {
            continue;
        }

        text += token_to_piece(token);
    }
    return text;
}

std::string Tokenizer::format_generation(const std::vector<int32_t> &tokens) const
{
    std::string text;
    for (const int32_t token : tokens)
    {
        if (token == bos_id_ || token == eos_id_)
        {
            continue;
        }

        text += piece_to_text(token_to_piece(token));
    }
    return text;
}

} // namespace tllm::tokenizer
