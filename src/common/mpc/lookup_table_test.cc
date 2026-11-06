// Copyright 2026 Ant International, Ant Group. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/common/mpc/lookup_table.h"

#include <cstdint>
#include <future>
#include <memory>
#include <random>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "gtest/gtest.h"
#include "src/common/mpc/elementwise.h"
#include "src/common/mpc/rot_context.h"
#include "src/common/runtime/link/mock_link_io.h"

namespace capricorn::operator_ {

class LookupTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lctxs_ = capricorn::SetupBrpcWorld("lookup_table_test", 2);
    thread_num_ = 4;
  }

  void TearDown() override {}

  std::vector<std::shared_ptr<yacl::link::Context>> lctxs_;
  uint32_t thread_num_;
};

// Test single table lookup
TEST_F(LookupTableTest, SingleTableLookupTest) {
  const uint32_t len = 10;
  const uint32_t input_bw = 4;  // 16 entries
  const uint32_t output_bw = 40;
  const uint32_t table_size = 1 << input_bw;
  const uint8_t index_mask = (input_bw == 8) ? -1 : (1u << input_bw) - 1;
  const uint64_t mod_mask = (output_bw == 64) ? -1ULL : (1ULL << output_bw) - 1;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;

  // Generate public table
  std::vector<uint64_t> public_table(table_size);
  prg.random_data(public_table.data(), table_size * sizeof(uint64_t));
  ElementwiseAndInplace(table_size, public_table.data(), mod_mask);

  // Generate secret-shared indices
  std::vector<uint8_t> server_idx(len);
  std::vector<uint8_t> client_idx(len);
  std::vector<uint8_t> original_indices(len);

  prg.random_data(server_idx.data(), len);
  prg.random_data(client_idx.data(), len);
  ElementwiseAndInplace(len, server_idx.data(), index_mask);
  ElementwiseAndInplace(len, client_idx.data(), index_mask);
  ElementwiseAddmod(len, server_idx.data(), client_idx.data(),
                    original_indices.data(), index_mask);

  std::vector<uint64_t> server_z(len);
  std::vector<uint64_t> client_z(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    LookupTableContext lut_ctx(kServerId, &rot_ctx, thread_num_);
    lut_ctx.LookupTable(len, server_idx.data(), server_z.data(), input_bw,
                        output_bw, public_table.data());
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    LookupTableContext lut_ctx(kClientId, &rot_ctx, thread_num_);
    lut_ctx.LookupTable(len, client_idx.data(), client_z.data(), input_bw,
                        output_bw, public_table.data());
  });

  server_task.get();
  client_task.get();

  // Verify: z = table[index]
  for (uint32_t i = 0; i < len; i++) {
    uint64_t expected = public_table[original_indices[i]];
    uint64_t computed = (server_z[i] + client_z[i]) & mod_mask;
    EXPECT_EQ(computed, expected)
        << "Single table lookup test failed at index " << i
        << ", original_index=" << static_cast<uint32_t>(original_indices[i])
        << ", expected=" << expected << ", computed=" << computed;
  }
}

// Test dual table lookup (LookupTable2)
TEST_F(LookupTableTest, DualTableLookupTest) {
  const uint32_t len = 10;
  const uint32_t input_bw = 4;  // 16 entries
  const uint32_t output_bw_x = 40;
  const uint32_t output_bw_y = 40;
  const uint32_t table_size = 1 << input_bw;
  const uint8_t index_mask = (input_bw == 8) ? -1u : ((1u << input_bw) - 1);
  const uint64_t x_mask =
      (output_bw_x == 64) ? -1ULL : ((1ULL << output_bw_x) - 1);
  const uint64_t y_mask =
      (output_bw_y == 64) ? -1ULL : ((1ULL << output_bw_y) - 1);

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;

  // Generate public tables
  std::vector<uint64_t> public_table_x(table_size);
  std::vector<uint64_t> public_table_y(table_size);
  prg.random_data(public_table_x.data(), table_size * sizeof(uint64_t));
  prg.random_data(public_table_y.data(), table_size * sizeof(uint64_t));
  ElementwiseAndInplace(table_size, public_table_x.data(), x_mask);
  ElementwiseAndInplace(table_size, public_table_y.data(), y_mask);

  // Generate secret-shared indices
  std::vector<uint8_t> server_idx(len);
  std::vector<uint8_t> client_idx(len);
  std::vector<uint8_t> original_indices(len);

  prg.random_data(server_idx.data(), len);
  prg.random_data(client_idx.data(), len);
  ElementwiseAndInplace(len, server_idx.data(), index_mask);
  ElementwiseAndInplace(len, client_idx.data(), index_mask);
  ElementwiseAddmod(len, server_idx.data(), client_idx.data(),
                    original_indices.data(), index_mask);

  std::vector<uint64_t> server_zx(len);
  std::vector<uint64_t> server_zy(len);
  std::vector<uint64_t> client_zx(len);
  std::vector<uint64_t> client_zy(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    LookupTableContext lut_ctx(kServerId, &rot_ctx, thread_num_);
    lut_ctx.LookupTable2(len, server_idx.data(), server_zx.data(),
                         server_zy.data(), input_bw, output_bw_x, output_bw_y,
                         public_table_x.data(), public_table_y.data());
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    LookupTableContext lut_ctx(kClientId, &rot_ctx, thread_num_);
    lut_ctx.LookupTable2(len, client_idx.data(), client_zx.data(),
                         client_zy.data(), input_bw, output_bw_x, output_bw_y,
                         public_table_x.data(), public_table_y.data());
  });

  server_task.get();
  client_task.get();

  // Verify: zx = table_x[index], zy = table_y[index]
  for (uint32_t i = 0; i < len; i++) {
    uint64_t expected_x = public_table_x[original_indices[i]];
    uint64_t expected_y = public_table_y[original_indices[i]];
    uint64_t computed_x = (server_zx[i] + client_zx[i]) & x_mask;
    uint64_t computed_y = (server_zy[i] + client_zy[i]) & y_mask;

    EXPECT_EQ(computed_x, expected_x)
        << "Dual table lookup (x) test failed at index " << i
        << ", original_index=" << static_cast<uint32_t>(original_indices[i]);
    EXPECT_EQ(computed_y, expected_y)
        << "Dual table lookup (y) test failed at index " << i
        << ", original_index=" << static_cast<uint32_t>(original_indices[i]);
  }
}

}  // namespace capricorn::operator_

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
