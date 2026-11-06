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

#include "src/common/mpc/truncate.h"

#include <cstdint>
#include <future>
#include <memory>
#include <random>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "gtest/gtest.h"
#include "src/common/mpc/and_protocol.h"
#include "src/common/mpc/b2a.h"
#include "src/common/mpc/compare.h"
#include "src/common/mpc/mux.h"
#include "src/common/mpc/rot_context.h"
#include "src/common/runtime/link/mock_link_io.h"

namespace capricorn::operator_ {

class TruncateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lctxs_ = capricorn::SetupBrpcWorld("truncate_test", 2);
    thread_num_ = 4;
  }

  void TearDown() override {}

  std::vector<std::shared_ptr<yacl::link::Context>> lctxs_;
  uint32_t thread_num_;
};

// Test approximate truncation
TEST_F(TruncateTest, TruncateApproximateTest) {
  const uint32_t len = 10;
  const uint32_t bw = 40;
  const uint32_t shift = 14;
  const uint64_t mod_mask = bw == 64 ? -1 : ((1ULL << bw) - 1);
  const uint64_t max_value = 1ULL << (bw - 2);

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;
  std::vector<uint64_t> server_x(len);
  std::vector<uint64_t> client_x(len);
  std::vector<int64_t> original_values(len);

  prg.random_data(original_values.data(), len * sizeof(uint64_t));
  for (uint32_t i = 0; i < len; i++) {
    original_values[i] = (original_values[i] % (2 * max_value)) - max_value;
  }

  prg.random_data(client_x.data(), len * sizeof(uint64_t));
  for (uint32_t i = 0; i < len; i++) {
    client_x[i] &= mod_mask;
    server_x[i] = (original_values[i] - client_x[i]) & mod_mask;
  }

  std::vector<uint64_t> server_z(len);
  std::vector<uint64_t> client_z(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    TruncateContext truncate_ctx(kServerId, &rot_ctx, thread_num_);
    truncate_ctx.TruncateApproximate(len, server_x.data(), server_z.data(), bw,
                                     shift);
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    TruncateContext truncate_ctx(kClientId, &rot_ctx, thread_num_);
    truncate_ctx.TruncateApproximate(len, client_x.data(), client_z.data(), bw,
                                     shift);
  });

  server_task.get();
  client_task.get();

  // Verify: approximate truncation allows error of at most 1
  for (uint32_t i = 0; i < len; i++) {
    int64_t expected = original_values[i] / static_cast<int64_t>(1ULL << shift);
    int64_t computed = GetSignedValue(server_z[i] + client_z[i], bw);
    EXPECT_LE(std::abs(computed - expected), 1)
        << "Truncate approximate test failed at index " << i
        << ", original=" << original_values[i] << ", expected=" << expected
        << ", computed=" << computed;
  }
}

// Test accurate truncation
TEST_F(TruncateTest, TruncateAccurateTest) {
  const uint32_t len = 10;
  const uint32_t bw = 40;
  const uint32_t shift = 14;
  const uint64_t mod_mask = bw == 64 ? -1 : ((1ULL << bw) - 1);
  const uint64_t max_value = 1ULL << (bw - 2);

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;
  std::vector<uint64_t> server_x(len);
  std::vector<uint64_t> client_x(len);
  std::vector<int64_t> original_values(len);

  prg.random_data(original_values.data(), len * sizeof(uint64_t));
  for (uint32_t i = 0; i < len; i++) {
    original_values[i] = std::abs(original_values[i]) % max_value;
  }

  prg.random_data(client_x.data(), len * sizeof(uint64_t));
  for (uint32_t i = 0; i < len; i++) {
    client_x[i] &= mod_mask;
    server_x[i] = (original_values[i] - client_x[i]) & mod_mask;
  }

  std::vector<uint64_t> server_z(len);
  std::vector<uint64_t> client_z(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    TruncateContext truncate_ctx(kServerId, &rot_ctx, thread_num_);
    truncate_ctx.TruncateAccurate(len, server_x.data(), server_z.data(), bw,
                                  shift);
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    TruncateContext truncate_ctx(kClientId, &rot_ctx, thread_num_);
    truncate_ctx.TruncateAccurate(len, client_x.data(), client_z.data(), bw,
                                  shift);
  });

  server_task.get();
  client_task.get();

  // Verify: accurate truncation should be exact
  for (uint32_t i = 0; i < len; i++) {
    int64_t expected = original_values[i] / static_cast<int64_t>(1ULL << shift);
    int64_t computed = GetSignedValue(server_z[i] + client_z[i], bw);
    EXPECT_EQ(computed, expected)
        << "Truncate accurate test failed at index " << i
        << ", original=" << original_values[i] << ", expected=" << expected
        << ", computed=" << computed;
  }
}

}  // namespace capricorn::operator_

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
