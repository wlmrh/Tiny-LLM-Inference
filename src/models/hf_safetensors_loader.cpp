#include "tiny_llm/models/hf_safetensors_loader.h"

#include "hf_json.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tiny_llm {

namespace {

constexpr size_t kSafeTensorHeaderPrefixBytes = 8;

uint64_t read_u64_little_endian(const std::vector<uint8_t>& bytes)
{
    if (bytes.size() < kSafeTensorHeaderPrefixBytes)
    {
        throw std::runtime_error("HFSafeTensorLoader::from_file: safetensor file is too small.");
    }

    uint64_t value = 0;
    for (size_t i = 0; i < kSafeTensorHeaderPrefixBytes; ++i)
    {
        value |= static_cast<uint64_t>(bytes[i]) << (8U * i);
    }
    return value;
}

std::vector<uint8_t> read_binary_file(const std::string& path)
{
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin)
    {
        throw std::runtime_error("HFSafeTensorLoader::from_file: failed to open file: " + path);
    }

    const std::streamsize file_size = fin.tellg();
    if (file_size <= 0)
    {
        throw std::runtime_error("HFSafeTensorLoader::from_file: file is empty: " + path);
    }

    fin.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(file_size), 0);
    if (!fin.read(reinterpret_cast<char*>(data.data()), file_size))
    {
        throw std::runtime_error("HFSafeTensorLoader::from_file: failed to read file content: " + path);
    }

    return data;
}

size_t checked_add(size_t lhs, size_t rhs, const std::string& error_prefix)
{
    if (rhs > std::numeric_limits<size_t>::max() - lhs)
    {
        throw std::runtime_error(error_prefix + ": size addition overflow.");
    }
    return lhs + rhs;
}

size_t checked_mul(size_t lhs, size_t rhs, const std::string& error_prefix)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        throw std::runtime_error(error_prefix + ": size multiplication overflow.");
    }
    return lhs * rhs;
}

struct ParsedDType {
    DType runtime_dtype = DType::kFloat32;
    size_t storage_bytes = sizeof(float);
};

ParsedDType parse_dtype(const std::string& dtype_token)
{
    if (dtype_token == "F32")
    {
        return ParsedDType{DType::kFloat32, sizeof(float)};
    }
    if (dtype_token == "F16" || dtype_token == "BF16")
    {
        return ParsedDType{DType::kFloat32, sizeof(uint16_t)};
    }

    throw std::runtime_error(
        "HFSafeTensorLoader::from_file: unsupported dtype token: " + dtype_token + ".");
}

size_t dtype_nbytes(const ParsedDType& dtype)
{
    return dtype.storage_bytes;
}

std::vector<int64_t> parse_shape(const hf_json::Value& value)
{
    const std::string err = "HFSafeTensorLoader::from_file";
    const auto& shape_values = value.as_array(err + ": shape must be array");

    std::vector<int64_t> shape;
    shape.reserve(shape_values.size());
    for (const hf_json::Value& dim_value : shape_values)
    {
        const int64_t dim = dim_value.as_int64(err + ": shape dimension must be integer");
        if (dim < 0)
        {
            throw std::runtime_error(err + ": shape dimension must be non-negative.");
        }
        shape.push_back(dim);
    }

    return shape;
}

std::pair<size_t, size_t> parse_offsets(const hf_json::Value& value)
{
    const std::string err = "HFSafeTensorLoader::from_file";
    const auto& offsets = value.as_array(err + ": data_offsets must be array");
    if (offsets.size() != 2)
    {
        throw std::runtime_error(err + ": data_offsets must contain exactly two numbers.");
    }

    const int64_t begin = offsets[0].as_int64(err + ": data_offsets[0] must be integer");
    const int64_t end = offsets[1].as_int64(err + ": data_offsets[1] must be integer");
    if (begin < 0 || end < 0)
    {
        throw std::runtime_error(err + ": data_offsets must be non-negative.");
    }
    if (end < begin)
    {
        throw std::runtime_error(err + ": data_offsets must satisfy end >= begin.");
    }

    return {static_cast<size_t>(begin), static_cast<size_t>(end)};
}

size_t tensor_nbytes_from_shape(const std::vector<int64_t>& shape, const ParsedDType& dtype)
{
    size_t element_count = 1;
    for (int64_t dim : shape)
    {
        element_count = checked_mul(
            element_count,
            static_cast<size_t>(dim),
            "HFSafeTensorLoader::from_file");
    }

    return checked_mul(element_count, dtype_nbytes(dtype), "HFSafeTensorLoader::from_file");
}

} // namespace

HFSafeTensorLoader HFSafeTensorLoader::from_file(const std::string& path)
{
    HFSafeTensorLoader loader;
    loader.file_path_ = path;
    loader.raw_file_ = read_binary_file(path);

    if (loader.raw_file_.size() < kSafeTensorHeaderPrefixBytes)
    {
        throw std::runtime_error("HFSafeTensorLoader::from_file: safetensor file is too small.");
    }

    const uint64_t header_length_u64 = read_u64_little_endian(loader.raw_file_);
    if (header_length_u64 == 0)
    {
        throw std::runtime_error("HFSafeTensorLoader::from_file: header length must be positive.");
    }

    const size_t header_length = static_cast<size_t>(header_length_u64);
    const size_t expected_data_start = checked_add(
        kSafeTensorHeaderPrefixBytes,
        header_length,
        "HFSafeTensorLoader::from_file");

    if (expected_data_start > loader.raw_file_.size())
    {
        throw std::runtime_error("HFSafeTensorLoader::from_file: header length exceeds file size.");
    }

    loader.data_base_offset_ = expected_data_start;

    const char* header_begin = reinterpret_cast<const char*>(
        loader.raw_file_.data() + kSafeTensorHeaderPrefixBytes);
    const std::string header_json(header_begin, header_begin + static_cast<std::ptrdiff_t>(header_length));

    const hf_json::Value root = hf_json::parse(header_json, "HFSafeTensorLoader::from_file");
    const auto& root_object = root.as_object("HFSafeTensorLoader::from_file: root must be object");

    for (const auto& item : root_object)
    {
        const std::string& tensor_key = item.first;
        if (tensor_key == "__metadata__")
        {
            continue;
        }

        if (item.second == nullptr)
        {
            throw std::runtime_error(
                "HFSafeTensorLoader::from_file: tensor descriptor pointer must be non-null.");
        }

        const hf_json::Value& tensor_descriptor = *item.second;

        (void)tensor_descriptor.as_object(
            "HFSafeTensorLoader::from_file: tensor descriptor must be object");

        const std::string dtype_token = hf_json::require_object_field(
            tensor_descriptor,
            "dtype",
            "HFSafeTensorLoader::from_file")
                                       .as_string("HFSafeTensorLoader::from_file: dtype must be string");
        const ParsedDType dtype = parse_dtype(dtype_token);

        const std::vector<int64_t> shape = parse_shape(
            hf_json::require_object_field(tensor_descriptor,
                                          "shape",
                                          "HFSafeTensorLoader::from_file"));

        const auto [offset_begin, offset_end] = parse_offsets(
            hf_json::require_object_field(tensor_descriptor,
                                          "data_offsets",
                                          "HFSafeTensorLoader::from_file"));

        const size_t expected_bytes = tensor_nbytes_from_shape(shape, dtype);
        const size_t actual_bytes = offset_end - offset_begin;
        if (expected_bytes != actual_bytes)
        {
            throw std::runtime_error(
                "HFSafeTensorLoader::from_file: byte size mismatches shape for key: " + tensor_key);
        }

        HFSafeTensorInfo info;
        info.shape = shape;
        info.dtype = dtype.runtime_dtype;
        info.storage_dtype = dtype_token;
        info.byte_offset = offset_begin;
        info.byte_size = actual_bytes;

        loader.tensor_infos_[tensor_key] = std::move(info);
    }

    if (loader.tensor_infos_.empty())
    {
        throw std::runtime_error("HFSafeTensorLoader::from_file: no tensor entries were found.");
    }

    struct TensorSpan {
        size_t begin = 0;
        size_t end = 0;
        std::string key;
    };

    std::vector<TensorSpan> spans;
    spans.reserve(loader.tensor_infos_.size());

    for (const auto& item : loader.tensor_infos_)
    {
        const std::string& key = item.first;
        const HFSafeTensorInfo& info = item.second;

        const size_t absolute_begin = checked_add(
            loader.data_base_offset_,
            info.byte_offset,
            "HFSafeTensorLoader::from_file");
        const size_t absolute_end = checked_add(
            absolute_begin,
            info.byte_size,
            "HFSafeTensorLoader::from_file");

        if (absolute_end > loader.raw_file_.size())
        {
            throw std::runtime_error(
                "HFSafeTensorLoader::from_file: tensor range exceeds file size for key: " + key);
        }

        spans.push_back(TensorSpan{absolute_begin, absolute_end, key});
    }

    std::sort(
        spans.begin(),
        spans.end(),
        [](const TensorSpan& lhs, const TensorSpan& rhs) {
            if (lhs.begin != rhs.begin)
            {
                return lhs.begin < rhs.begin;
            }
            return lhs.end < rhs.end;
        });

    for (size_t i = 1; i < spans.size(); ++i)
    {
        if (spans[i].begin < spans[i - 1].end)
        {
            throw std::runtime_error(
                "HFSafeTensorLoader::from_file: overlapping tensor ranges detected: "
                + spans[i - 1].key + " and " + spans[i].key);
        }
    }

    return loader;
}

bool HFSafeTensorLoader::has_tensor(const std::string& key) const
{
    return tensor_infos_.find(key) != tensor_infos_.end();
}

Tensor HFSafeTensorLoader::tensor(const std::string& key) const
{
    const HFSafeTensorInfo& info = require_tensor_info(key);

    const auto options = torch::TensorOptions()
        .dtype(to_torch_scalar_type(info.dtype))
        .device(c10::kCPU);

    if (info.byte_size == 0)
    {
        return torch::empty(info.shape, options);
    }

    const size_t absolute_offset = data_base_offset_ + info.byte_offset;
    const uint8_t* data_ptr = raw_file_.data() + absolute_offset;

    if (info.storage_dtype == "BF16")
    {
        const auto source_options = torch::TensorOptions().dtype(torch::kBFloat16).device(c10::kCPU);
        Tensor source = torch::from_blob(const_cast<uint8_t*>(data_ptr), info.shape, source_options);
        return source.to(torch::kFloat32).contiguous();
    }

    if (info.storage_dtype == "F16")
    {
        const auto source_options = torch::TensorOptions().dtype(torch::kFloat16).device(c10::kCPU);
        Tensor source = torch::from_blob(const_cast<uint8_t*>(data_ptr), info.shape, source_options);
        return source.to(torch::kFloat32).contiguous();
    }

    if (info.storage_dtype == "F32"
        && (reinterpret_cast<uintptr_t>(data_ptr) % alignof(float)) != 0)
    {
        Tensor aligned_tensor = torch::empty(info.shape, options);
        std::memcpy(aligned_tensor.data_ptr(), data_ptr, info.byte_size);
        return aligned_tensor;
    }

    return make_tensor_from_blob(
        const_cast<uint8_t*>(data_ptr),
        info.shape,
        info.dtype);
}

std::vector<int64_t> HFSafeTensorLoader::shape(const std::string& key) const
{
    return require_tensor_info(key).shape;
}

DType HFSafeTensorLoader::dtype(const std::string& key) const
{
    return require_tensor_info(key).dtype;
}

std::vector<std::string> HFSafeTensorLoader::keys() const
{
    std::vector<std::string> out;
    out.reserve(tensor_infos_.size());
    for (const auto& item : tensor_infos_)
    {
        out.push_back(item.first);
    }

    std::sort(out.begin(), out.end());
    return out;
}

const HFSafeTensorInfo& HFSafeTensorLoader::require_tensor_info(const std::string& key) const
{
    const auto it = tensor_infos_.find(key);
    if (it == tensor_infos_.end())
    {
        throw std::runtime_error("HFSafeTensorLoader: tensor key does not exist: " + key);
    }
    return it->second;
}

} // namespace tiny_llm
