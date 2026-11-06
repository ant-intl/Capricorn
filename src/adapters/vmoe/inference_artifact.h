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

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace capricorn::vmoe {

enum class InputVisibility { kPublic, kSecret, kPrivate };

struct ManifestEntry {
  std::filesystem::path npy_path;
  std::string symbol_name;
  InputVisibility visibility;
  size_t owner_rank;
};

struct Float32Tensor {
  std::vector<int64_t> shape;
  std::vector<float> values;
};

std::vector<ManifestEntry> LoadInputManifest(
    const std::filesystem::path& artifact_dir);

Float32Tensor LoadFloat32Npy(const std::filesystem::path& path);

std::vector<int64_t> LoadExpectedOutputShape(
    const std::filesystem::path& artifact_dir);

void SaveFloat32Npy(const std::filesystem::path& path,
                    const std::vector<int64_t>& shape,
                    const std::vector<float>& values);

}  // namespace capricorn::vmoe
