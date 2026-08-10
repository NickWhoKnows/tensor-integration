#include "tllm/gguf/loader.h"

#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tllm::gguf
{
namespace
{

enum MetadataType : uint32_t
{
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

size_t pad_offset(size_t offset, size_t alignment)
{
    return (offset + alignment - 1) & ~(alignment - 1);
}

} // namespace

Loader::Loader(const std::string &path) : path_(path)
{
    ctx_ = ggml_init(ctx_params_);
    if (ctx_ == nullptr)
    {
        throw std::runtime_error("failed to initialize ggml context");
    }

    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
    {
        throw std::runtime_error("failed to open file: " + path);
    }

    struct stat st {};
    if (fstat(fd_, &st) != 0)
    {
        throw std::runtime_error("failed to stat file: " + path);
    }

    file_size_ = static_cast<size_t>(st.st_size);
    mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_data_ == MAP_FAILED)
    {
        throw std::runtime_error("failed to mmap file: " + path);
    }

    const uint32_t magic = read_scalar<uint32_t>();
    if (magic != 0x46554747)
    {
        throw std::runtime_error("invalid GGUF magic in file: " + path);
    }

    const uint32_t version = read_scalar<uint32_t>();
    if (version < 2 || version > 3)
    {
        throw std::runtime_error("unsupported GGUF version: " + std::to_string(version));
    }

    const uint64_t tensor_count = read_scalar<uint64_t>();
    const uint64_t metadata_count = read_scalar<uint64_t>();

    metadata_.reserve(static_cast<size_t>(metadata_count));
    for (uint64_t i = 0; i < metadata_count; ++i)
    {
        Metadata entry;
        entry.key = read_string();
        entry.type = read_scalar<uint32_t>();
        entry.value = read_metadata_value(entry.type);
        metadata_.push_back(std::move(entry));
    }

    tensors_.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i)
    {
        TensorInfo tensor;
        tensor.name = read_string();
        const uint32_t n_dims = read_scalar<uint32_t>();
        tensor.dimensions.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d)
        {
            tensor.dimensions[d] = read_scalar<int64_t>();
        }
        tensor.type = static_cast<ggml_type>(read_scalar<uint32_t>());
        tensor.offset = read_scalar<uint64_t>();
        tensor.nbytes = tensor_byte_size(tensor);
        tensors_.push_back(std::move(tensor));
    }

    alignment_ = file_alignment();
    if (!tensors_.empty())
    {
        tensor_data_offset_ = pad_offset(cursor_, alignment_);
    }

    bind_tensor_data();
}

Loader::~Loader()
{
    if (ctx_ != nullptr)
    {
        ggml_free(ctx_);
    }

    if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED)
    {
        munmap(mapped_data_, file_size_);
    }

    if (fd_ >= 0)
    {
        close(fd_);
    }
}

MetaValue Loader::read_metadata_value(const uint32_t type)
{
    switch (type)
    {
    case UINT8:
        return read_scalar<uint8_t>();
    case INT8:
        return read_scalar<int8_t>();
    case UINT16:
        return read_scalar<uint16_t>();
    case INT16:
        return read_scalar<int16_t>();
    case UINT32:
        return read_scalar<uint32_t>();
    case INT32:
        return read_scalar<int32_t>();
    case FLOAT32:
        return read_scalar<float>();
    case BOOL:
        return static_cast<bool>(read_scalar<uint8_t>());
    case STRING:
        return read_string();
    case ARRAY:
    {
        const uint32_t elem_type = read_scalar<uint32_t>();
        const uint64_t count = read_scalar<uint64_t>();
        return read_array_value(elem_type, count);
    }
    case UINT64:
        return read_scalar<uint64_t>();
    case INT64:
        return read_scalar<int64_t>();
    case FLOAT64:
        return read_scalar<double>();
    default:
        throw std::runtime_error("invalid metadata value type");
    }
}

MetaValue Loader::read_array_value(const uint32_t elem_type, const uint64_t count)
{
    switch (elem_type)
    {
    case UINT8:
    {
        std::vector<uint8_t> values;
        values.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
        {
            values.push_back(read_scalar<uint8_t>());
        }
        return values;
    }
    case INT8:
    {
        std::vector<int8_t> values;
        values.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
        {
            values.push_back(read_scalar<int8_t>());
        }
        return values;
    }
    case UINT16:
    {
        std::vector<uint16_t> values;
        values.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
        {
            values.push_back(read_scalar<uint16_t>());
        }
        return values;
    }
    case INT16:
    {
        std::vector<int16_t> values;
        values.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
        {
            values.push_back(read_scalar<int16_t>());
        }
        return values;
    }
    case UINT32:
    {
        std::vector<uint32_t> values;
        values.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
        {
            values.push_back(read_scalar<uint32_t>());
        }
        return values;
    }
    case INT32:
    {
        std::vector<int32_t> values;
        values.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
        {
            values.push_back(read_scalar<int32_t>());
        }
        return values;
    }
    case FLOAT32:
    {
        std::vector<float> values;
        values.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
        {
            values.push_back(read_scalar<float>());
        }
        return values;
    }
    case BOOL:
    {
        std::vector<bool> values;
        values.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
        {
            values.push_back(static_cast<bool>(read_scalar<uint8_t>()));
        }
        return values;
    }
    case STRING:
    {
        std::vector<std::string> values;
        values.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
        {
            values.push_back(read_string());
        }
        return values;
    }
    case UINT64:
    {
        std::vector<uint64_t> values;
        values.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
        {
            values.push_back(read_scalar<uint64_t>());
        }
        return values;
    }
    case INT64:
    {
        std::vector<int64_t> values;
        values.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
        {
            values.push_back(read_scalar<int64_t>());
        }
        return values;
    }
    case FLOAT64:
    {
        std::vector<double> values;
        values.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
        {
            values.push_back(read_scalar<double>());
        }
        return values;
    }
    case ARRAY:
        throw std::runtime_error("nested array metadata not supported");
    default:
        throw std::runtime_error("invalid array element type");
    }
}

std::string Loader::read_string()
{
    const uint64_t len = read_scalar<uint64_t>();
    if (cursor_ + len > file_size_)
    {
        throw std::runtime_error("string extends past end of GGUF file");
    }

    std::string str(reinterpret_cast<char *>(static_cast<char *>(mapped_data_) + cursor_), len);
    cursor_ += len;
    return str;
}

uint32_t Loader::file_alignment() const
{
    for (const auto &entry : metadata_)
    {
        if (entry.key == "general.alignment")
        {
            if (const auto *value = std::get_if<uint32_t>(&entry.value))
            {
                return *value;
            }
        }
    }
    return 32;
}

size_t Loader::tensor_byte_size(const TensorInfo &tensor) const
{
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (size_t i = 0; i < tensor.dimensions.size() && i < GGML_MAX_DIMS; ++i)
    {
        ne[i] = tensor.dimensions[i];
    }

    const auto type = static_cast<enum ::ggml_type>(tensor.type);
    const int64_t block_size = ggml_blck_size(type);
    if (block_size == 0 || ne[0] % block_size != 0)
    {
        throw std::runtime_error("invalid tensor shape for type: " + tensor.name);
    }

    int64_t nblocks = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i)
    {
        nblocks *= ne[i];
    }
    nblocks /= block_size;

    return static_cast<size_t>(nblocks) * ggml_type_size(type);
}

void Loader::bind_tensor_data()
{
    for (auto &tensor : tensors_)
    {
        const size_t file_offset = tensor_data_offset_ + tensor.offset;
        if (file_offset > file_size_ || tensor.nbytes > file_size_ - file_offset)
        {
            throw std::runtime_error("tensor data out of bounds: " + tensor.name);
        }
        tensor.data = static_cast<const char *>(mapped_data_) + file_offset;
    }
}

const TensorInfo *Loader::find_tensor(const std::string &name) const
{
    for (const auto &tensor : tensors_)
    {
        if (tensor.name == name)
        {
            return &tensor;
        }
    }
    return nullptr;
}

const TensorInfo &Loader::tensor(const std::string &name) const
{
    const TensorInfo *found = find_tensor(name);
    if (found == nullptr)
    {
        throw std::runtime_error("tensor not found: " + name);
    }
    return *found;
}

const MetaValue *Loader::find_metadata(const std::string &key) const
{
    for (const auto &entry : metadata_)
    {
        if (entry.key == key)
        {
            return &entry.value;
        }
    }
    return nullptr;
}

ggml_tensor *Loader::wrap_tensor(const std::string &name)
{
    const TensorInfo *info = find_tensor(name);
    if (info == nullptr)
    {
        throw std::runtime_error("tensor not found: " + name);
    }

    ggml_tensor *tensor = nullptr;
    switch (info->dimensions.size())
    {
    case 1:
        tensor = ggml_new_tensor_1d(ctx_, info->type, info->dimensions[0]);
        break;
    case 2:
        tensor = ggml_new_tensor_2d(ctx_, info->type, info->dimensions[0], info->dimensions[1]);
        break;
    case 3:
        tensor = ggml_new_tensor_3d(ctx_, info->type, info->dimensions[0], info->dimensions[1], info->dimensions[2]);
        break;
    case 4:
        tensor = ggml_new_tensor_4d(ctx_, info->type, info->dimensions[0], info->dimensions[1], info->dimensions[2], info->dimensions[3]);
        break;
    default:
        tensor = ggml_new_tensor(ctx_, info->type, static_cast<int>(info->dimensions.size()), info->dimensions.data());
        break;
    }

    tensor->data = const_cast<void *>(info->data);
    ggml_set_name(tensor, info->name.c_str());
    return tensor;
}

int64_t Loader::total_parameters() const
{
    int64_t total = 0;
    for (const auto &tensor : tensors_)
    {
        int64_t count = 1;
        for (const auto dim : tensor.dimensions)
        {
            count *= dim;
        }
        total += count;
    }
    return total;
}

void Loader::print_tensors() const
{
    std::cout << "Tensors:" << std::endl;
    for (const auto &tensor : tensors_)
    {
        std::cout << "  " << tensor.name << " [";
        for (size_t i = 0; i < tensor.dimensions.size(); ++i)
        {
            std::cout << tensor.dimensions[i];
            if (i + 1 < tensor.dimensions.size())
            {
                std::cout << ", ";
            }
        }
        std::cout << "] " << ggml_type_name(tensor.type) << std::endl;
    }
}

void Loader::print_metadata() const
{
    std::cout << "Metadata:" << std::endl;
    for (const auto &entry : metadata_)
    {
        std::cout << "  " << entry.key << std::endl;
    }
}


} // namespace tllm::gguf
