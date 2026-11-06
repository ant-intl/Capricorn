// Copyright 2026 Ant International, Ant Group. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/adapters/vmoe/inference_artifact.h"

#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "yacl/base/exception.h"

namespace capricorn::vmoe {
namespace {

constexpr char kNpyMagic[] = "\x93NUMPY";

uint16_t ReadLittleEndian16(const std::vector<char>& bytes, size_t offset) {
  return static_cast<uint16_t>(static_cast<unsigned char>(bytes[offset])) |
         static_cast<uint16_t>(static_cast<unsigned char>(bytes[offset + 1]))
             << 8;
}

uint32_t ReadLittleEndian32(const std::vector<char>& bytes, size_t offset) {
  return static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset])) |
         static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 1]))
             << 8 |
         static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 2]))
             << 16 |
         static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 3]))
             << 24;
}

std::vector<char> ReadFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  YACL_ENFORCE(input.is_open(), "Unable to open {}", path.string());
  const auto end = input.tellg();
  YACL_ENFORCE(end >= 0, "Unable to determine size of {}", path.string());
  std::vector<char> bytes(static_cast<size_t>(end));
  input.seekg(0);
  input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  YACL_ENFORCE(input.good() || input.eof(), "Unable to read {}", path.string());
  return bytes;
}

std::string RegexGroup(const std::string& header, const std::regex& pattern,
                       std::string_view field) {
  std::smatch match;
  YACL_ENFORCE(std::regex_search(header, match, pattern),
               "NPY header is missing {}", field);
  return match[1].str();
}

std::vector<int64_t> ParseShape(const std::string& shape_text) {
  std::vector<int64_t> shape;
  std::istringstream dimensions(shape_text);
  std::string item;
  while (std::getline(dimensions, item, ',')) {
    const auto first = item.find_first_not_of(" \t");
    if (first == std::string::npos) {
      continue;
    }
    const auto last = item.find_last_not_of(" \t");
    const auto value_text = item.substr(first, last - first + 1);
    size_t parsed = 0;
    int64_t value = 0;
    try {
      value = std::stoll(value_text, &parsed);
    } catch (const std::exception&) {
      YACL_THROW("Invalid NPY shape dimension {}", value_text);
    }
    YACL_ENFORCE(parsed == value_text.size() && value > 0,
                 "NPY dimensions must be positive integers, got {}",
                 value_text);
    shape.push_back(value);
  }
  YACL_ENFORCE(!shape.empty(), "NPY tensor rank must be positive");
  return shape;
}

size_t ElementCount(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (const auto dimension : shape) {
    YACL_ENFORCE(dimension > 0, "Tensor dimensions must be positive");
    const auto size = static_cast<size_t>(dimension);
    YACL_ENFORCE(count <= std::numeric_limits<size_t>::max() / size,
                 "Tensor element count overflows size_t");
    count *= size;
  }
  return count;
}

bool IsLittleEndianHost() {
  const uint16_t marker = 1;
  return *reinterpret_cast<const unsigned char*>(&marker) == 1;
}

}  // namespace

Float32Tensor LoadFloat32Npy(const std::filesystem::path& path) {
  const auto bytes = ReadFile(path);
  YACL_ENFORCE(bytes.size() >= 10, "NPY file is too short: {}", path.string());
  YACL_ENFORCE(std::memcmp(bytes.data(), kNpyMagic, 6) == 0,
               "Invalid NPY magic in {}", path.string());

  const auto major = static_cast<unsigned char>(bytes[6]);
  const auto minor = static_cast<unsigned char>(bytes[7]);
  YACL_ENFORCE(minor == 0 && (major == 1 || major == 2),
               "Only NPY versions 1.0 and 2.0 are supported, got {}.{}", major,
               minor);
  const size_t length_size = major == 1 ? 2 : 4;
  YACL_ENFORCE(bytes.size() >= 8 + length_size, "Truncated NPY header length");
  const size_t header_size =
      major == 1 ? ReadLittleEndian16(bytes, 8) : ReadLittleEndian32(bytes, 8);
  const size_t header_offset = 8 + length_size;
  YACL_ENFORCE(header_size <= bytes.size() - header_offset,
               "Truncated NPY header");
  const std::string header(bytes.data() + header_offset, header_size);

  const auto descriptor = RegexGroup(
      header, std::regex("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]"),
      "descr");
  YACL_ENFORCE(descriptor == "<f4" || descriptor == "|f4",
               "Only little-endian float32 NPY arrays are supported, got {}",
               descriptor);
  YACL_ENFORCE(IsLittleEndianHost(),
               "Float32 NPY loading currently requires a little-endian host");
  const auto fortran = RegexGroup(
      header, std::regex("['\"]fortran_order['\"]\\s*:\\s*(True|False)"),
      "fortran_order");
  YACL_ENFORCE(fortran == "False", "Fortran-order NPY arrays are unsupported");
  const auto shape = ParseShape(RegexGroup(
      header, std::regex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)"), "shape"));
  const auto element_count = ElementCount(shape);
  YACL_ENFORCE(
      element_count <= std::numeric_limits<size_t>::max() / sizeof(float),
      "NPY payload size overflows size_t");
  const size_t payload_size = element_count * sizeof(float);
  const size_t payload_offset = header_offset + header_size;
  YACL_ENFORCE(bytes.size() - payload_offset == payload_size,
               "NPY payload size mismatch: expected {}, got {}", payload_size,
               bytes.size() - payload_offset);

  std::vector<float> values(element_count);
  std::memcpy(values.data(), bytes.data() + payload_offset, payload_size);
  return {shape, std::move(values)};
}

std::vector<int64_t> LoadExpectedOutputShape(
    const std::filesystem::path& artifact_dir) {
  return LoadFloat32Npy(artifact_dir / "expected.npy").shape;
}

void SaveFloat32Npy(const std::filesystem::path& path,
                    const std::vector<int64_t>& shape,
                    const std::vector<float>& values) {
  YACL_ENFORCE(!shape.empty(), "NPY tensor rank must be positive");
  YACL_ENFORCE(values.size() == ElementCount(shape),
               "Shape requires {} values, got {}", ElementCount(shape),
               values.size());
  YACL_ENFORCE(IsLittleEndianHost(),
               "Float32 NPY saving currently requires a little-endian host");

  std::ostringstream shape_stream;
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      shape_stream << ", ";
    }
    shape_stream << shape[index];
  }
  if (shape.size() == 1) {
    shape_stream << ',';
  }
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                       shape_stream.str() + "), }";
  constexpr size_t prefix_size = 6 + 2 + 2;
  const size_t padding = 16 - ((prefix_size + header.size() + 1) % 16);
  header.append(padding, ' ');
  header.push_back('\n');
  YACL_ENFORCE(header.size() <= std::numeric_limits<uint16_t>::max(),
               "NPY v1.0 header is too large");

  std::ofstream output(path, std::ios::binary);
  YACL_ENFORCE(output.is_open(), "Unable to open {} for writing",
               path.string());
  output.write(kNpyMagic, 6);
  output.put(1);
  output.put(0);
  const auto header_size = static_cast<uint16_t>(header.size());
  output.put(static_cast<char>(header_size & 0xff));
  output.put(static_cast<char>((header_size >> 8) & 0xff));
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  YACL_ENFORCE(output.good(), "Unable to write {}", path.string());
}

std::vector<ManifestEntry> LoadInputManifest(
    const std::filesystem::path& artifact_dir) {
  const auto manifest_path = artifact_dir / "input_order.txt";
  std::ifstream manifest(manifest_path);
  YACL_ENFORCE(manifest.is_open(), "Unable to open {}", manifest_path.string());

  std::vector<ManifestEntry> entries;
  std::unordered_set<std::string> symbol_names;
  std::string line;
  size_t line_number = 0;
  while (std::getline(manifest, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r") == std::string::npos) {
      continue;
    }
    std::istringstream fields(line);
    std::string relative_path_text;
    std::string symbol_name;
    std::string visibility_text;
    std::string owner_text;
    std::string extra;
    YACL_ENFORCE(fields >> relative_path_text >> symbol_name >>
                     visibility_text >> owner_text,
                 "Malformed manifest line {}", line_number);
    YACL_ENFORCE(!(fields >> extra), "Manifest line {} has extra fields",
                 line_number);

    const std::filesystem::path relative_path(relative_path_text);
    YACL_ENFORCE(!relative_path.empty() && !relative_path.is_absolute(),
                 "Manifest path must be relative on line {}", line_number);
    for (const auto& component : relative_path) {
      YACL_ENFORCE(component != "..",
                   "Manifest path traversal is forbidden on line {}",
                   line_number);
    }
    YACL_ENFORCE(symbol_names.insert(symbol_name).second,
                 "Duplicate runtime symbol {}", symbol_name);

    InputVisibility visibility;
    if (visibility_text == "PUBLIC") {
      visibility = InputVisibility::kPublic;
    } else if (visibility_text == "SECRET") {
      visibility = InputVisibility::kSecret;
    } else if (visibility_text == "PRIVATE") {
      visibility = InputVisibility::kPrivate;
    } else {
      YACL_THROW("Unknown visibility {} on line {}", visibility_text,
                 line_number);
    }
    YACL_ENFORCE(owner_text == "0" || owner_text == "1",
                 "Owner rank must be 0 or 1 on line {}", line_number);
    entries.push_back({artifact_dir / relative_path, std::move(symbol_name),
                       visibility, owner_text == "0" ? 0U : 1U});
  }
  YACL_ENFORCE(!entries.empty(), "Input manifest must not be empty");
  return entries;
}

}  // namespace capricorn::vmoe
