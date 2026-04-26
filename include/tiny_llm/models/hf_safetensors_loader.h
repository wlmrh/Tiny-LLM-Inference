#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tiny_llm/core/tensor.h"

namespace tiny_llm {

struct HFSafeTensorInfo {
    std::vector<int64_t> shape;
    DType dtype = DType::kFloat32;
    std::string storage_dtype = "F32";
    size_t byte_offset = 0;
    size_t byte_size = 0;
};

class HFSafeTensorLoader {
public:
    static HFSafeTensorLoader from_file(const std::string& path);

    bool has_tensor(const std::string& key) const;
    Tensor tensor(const std::string& key) const;
    std::vector<int64_t> shape(const std::string& key) const;
    DType dtype(const std::string& key) const;
    std::vector<std::string> keys() const;

    const std::string& file_path() const { return file_path_; }

private:
    const HFSafeTensorInfo& require_tensor_info(const std::string& key) const;

    std::string file_path_;
    std::vector<uint8_t> raw_file_;
    size_t data_base_offset_ = 0;
    std::unordered_map<std::string, HFSafeTensorInfo> tensor_infos_;
};

} // namespace tiny_llm
