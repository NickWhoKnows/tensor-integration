#include "tllm/tokenizer/chat_template.h"

#include <cctype>
#include <stdexcept>

namespace tllm::tokenizer
{
namespace
{

constexpr const char kBeginOfText[] = "<|begin_of_text|>";
constexpr const char kStartHeaderId[] = "<|start_header_id|>";
constexpr const char kEndHeaderId[] = "<|end_header_id|>";
constexpr const char kEotId[] = "<|eot_id|>";

std::string trim(const std::string &text)
{
    const auto is_space = [](const unsigned char c) { return std::isspace(c) != 0; };
    size_t begin = 0;
    while (begin < text.size() && is_space(static_cast<unsigned char>(text[begin])))
    {
        ++begin;
    }

    size_t end = text.size();
    while (end > begin && is_space(static_cast<unsigned char>(text[end - 1])))
    {
        --end;
    }

    return text.substr(begin, end - begin);
}

void push_segment(std::vector<ChatSegment> &segments, const bool literal, const std::string &value)
{
    if (!literal && value.empty())
    {
        return;
    }

    segments.emplace_back(literal, value);
}

void push_role_header(std::vector<ChatSegment> &segments, const std::string &role)
{
    push_segment(segments, true, kStartHeaderId);
    push_segment(segments, false, role);
    push_segment(segments, true, kEndHeaderId);
    push_segment(segments, false, "\n\n");
}

} // namespace

std::vector<ChatSegment> llama3_chat_segments(const std::vector<ChatMessage> &messages,
                                              const ChatTemplateOptions &options)
{
    if (messages.empty())
    {
        throw std::runtime_error("chat template requires at least one message");
    }

    std::string system_message;
    std::vector<ChatMessage> conversation = messages;
    if (messages.front().role == "system")
    {
        system_message = trim(messages.front().content);
        conversation.assign(messages.begin() + 1, messages.end());
    }

    std::vector<ChatSegment> segments;
    push_segment(segments, true, kBeginOfText);
    push_role_header(segments, "system");
    push_segment(segments, false, "Cutting Knowledge Date: December 2023\nToday Date: " + options.date_string + "\n\n");
    push_segment(segments, false, system_message);
    push_segment(segments, true, kEotId);

    for (const ChatMessage &message : conversation)
    {
        if (message.role == "system")
        {
            throw std::runtime_error("only the first message may use role=system");
        }

        push_role_header(segments, message.role);
        push_segment(segments, false, trim(message.content));
        push_segment(segments, true, kEotId);
    }

    if (options.add_generation_prompt)
    {
        push_role_header(segments, "assistant");
    }

    return segments;
}

} // namespace tllm::tokenizer
