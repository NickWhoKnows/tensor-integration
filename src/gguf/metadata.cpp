#include "tllm/gguf/metadata.h"

#include <stdexcept>

namespace tllm::gguf::meta
{
namespace
{

template <typename T>
const T *as_scalar(const MetaValue *value)
{
    return value != nullptr ? std::get_if<T>(value) : nullptr;
}

} // namespace

std::optional<std::string> read_string(const Loader &loader, const std::string &key)
{
    const MetaValue *value = loader.find_metadata(key);
    if (value == nullptr)
    {
        return std::nullopt;
    }

    if (const auto *str = as_scalar<std::string>(value))
    {
        return *str;
    }

    throw std::runtime_error("metadata key has unsupported string type: " + key);
}

std::optional<int32_t> read_int(const Loader &loader, const std::string &key)
{
    const MetaValue *value = loader.find_metadata(key);
    if (value == nullptr)
    {
        return std::nullopt;
    }

    if (const auto *v = as_scalar<uint32_t>(value))
    {
        return static_cast<int32_t>(*v);
    }
    if (const auto *v = as_scalar<int32_t>(value))
    {
        return *v;
    }
    if (const auto *v = as_scalar<uint64_t>(value))
    {
        return static_cast<int32_t>(*v);
    }
    if (const auto *v = as_scalar<int64_t>(value))
    {
        return static_cast<int32_t>(*v);
    }

    throw std::runtime_error("metadata key has unsupported integer type: " + key);
}

std::vector<std::string> read_string_array(const Loader &loader, const std::string &key)
{
    const MetaValue *value = loader.find_metadata(key);
    if (value == nullptr)
    {
        throw std::runtime_error("missing metadata array: " + key);
    }

    if (const auto *arr = as_scalar<std::vector<std::string>>(value))
    {
        return *arr;
    }

    throw std::runtime_error("metadata key is not a string array: " + key);
}

} // namespace tllm::gguf::meta
