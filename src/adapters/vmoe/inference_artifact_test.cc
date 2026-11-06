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

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace capricorn::vmoe {
namespace {

std::filesystem::path TestDir() {
  auto path = std::filesystem::path(testing::TempDir()) /
              testing::UnitTest::GetInstance()->current_test_info()->name();
  std::filesystem::create_directories(path);
  return path;
}

void WriteNpy(const std::filesystem::path& path, std::string_view descriptor,
              bool fortran_order, bool truncate_payload = false) {
  std::string header =
      "{'descr': '" + std::string(descriptor) +
      "', 'fortran_order': " + (fortran_order ? "True" : "False") +
      ", 'shape': (2, 3), }";
  const size_t prefix_size = 6 + 2 + 2;
  const size_t padding = 16 - ((prefix_size + header.size() + 1) % 16);
  header.append(padding, ' ');
  header.push_back('\n');

  std::ofstream output(path, std::ios::binary);
  output.write("\x93NUMPY", 6);
  output.put(1);
  output.put(0);
  const auto header_size = static_cast<uint16_t>(header.size());
  output.put(static_cast<char>(header_size & 0xff));
  output.put(static_cast<char>((header_size >> 8) & 0xff));
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  const std::vector<float> values = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const size_t value_count =
      truncate_payload ? values.size() - 1 : values.size();
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(value_count * sizeof(float)));
}

TEST(InferenceArtifactTest, LoadsAndSavesLittleEndianFloat32Npy) {
  const auto directory = TestDir();
  const auto input_path = directory / "input.npy";
  WriteNpy(input_path, "<f4", false);

  const auto tensor = LoadFloat32Npy(input_path);
  EXPECT_EQ(tensor.shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(tensor.values,
            (std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));

  const auto output_path = directory / "output.npy";
  SaveFloat32Npy(output_path, tensor.shape, tensor.values);
  const auto round_trip = LoadFloat32Npy(output_path);
  EXPECT_EQ(round_trip.shape, tensor.shape);
  EXPECT_EQ(round_trip.values, tensor.values);
}

TEST(InferenceArtifactTest, LoadsExpectedOutputShapeFromArtifact) {
  const auto directory = TestDir();
  SaveFloat32Npy(directory / "expected.npy", {1, 1000},
                 std::vector<float>(1000, 0.0F));

  EXPECT_EQ(LoadExpectedOutputShape(directory),
            (std::vector<int64_t>{1, 1000}));
}

TEST(InferenceArtifactTest, LoadsOrderedManifest) {
  const auto directory = TestDir();
  std::ofstream manifest(directory / "input_order.txt");
  manifest << "arg_000.npy token_ids PUBLIC 0\n"
           << "arg_001.npy token_embeddings SECRET 0\n"
           << "arg_002.npy router_weights PRIVATE 1\n"
           << "arg_003.npy expert_w1 PRIVATE 1\n"
           << "arg_004.npy expert_w2 PRIVATE 1\n";
  manifest.close();

  const auto entries = LoadInputManifest(directory);
  ASSERT_EQ(entries.size(), 5);
  EXPECT_EQ(entries[0].npy_path, directory / "arg_000.npy");
  EXPECT_EQ(entries[0].symbol_name, "token_ids");
  EXPECT_EQ(entries[0].visibility, InputVisibility::kPublic);
  EXPECT_EQ(entries[0].owner_rank, 0);
  EXPECT_EQ(entries[1].visibility, InputVisibility::kSecret);
  EXPECT_EQ(entries[4].symbol_name, "expert_w2");
  EXPECT_EQ(entries[4].visibility, InputVisibility::kPrivate);
  EXPECT_EQ(entries[4].owner_rank, 1);
}

TEST(InferenceArtifactTest, RejectsUnsupportedOrMalformedNpy) {
  const auto directory = TestDir();
  WriteNpy(directory / "wrong_dtype.npy", "<f8", false);
  WriteNpy(directory / "fortran.npy", "<f4", true);
  WriteNpy(directory / "truncated.npy", "<f4", false, true);

  EXPECT_ANY_THROW(LoadFloat32Npy(directory / "wrong_dtype.npy"));
  EXPECT_ANY_THROW(LoadFloat32Npy(directory / "fortran.npy"));
  EXPECT_ANY_THROW(LoadFloat32Npy(directory / "truncated.npy"));
}

TEST(InferenceArtifactTest, RejectsUnsafeOrAmbiguousManifest) {
  const auto directory = TestDir();
  const auto manifest_path = directory / "input_order.txt";

  {
    std::ofstream manifest(manifest_path);
    manifest << "../arg.npy tokens SECRET 0\n";
  }
  EXPECT_ANY_THROW(LoadInputManifest(directory));

  {
    std::ofstream manifest(manifest_path);
    manifest << "arg.npy tokens SECRET 2\n";
  }
  EXPECT_ANY_THROW(LoadInputManifest(directory));

  {
    std::ofstream manifest(manifest_path);
    manifest << "arg0.npy tokens SECRET 0\n"
             << "arg1.npy tokens PRIVATE 1\n";
  }
  EXPECT_ANY_THROW(LoadInputManifest(directory));
}

}  // namespace
}  // namespace capricorn::vmoe
