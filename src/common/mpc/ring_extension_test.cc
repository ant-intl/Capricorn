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

#include "src/common/mpc/ring_extension.h"

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

class RingExtensionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lctxs_ = capricorn::SetupBrpcWorld("ring_extension_test", 2);
    thread_num_ = 4;
  }

  void TearDown() override {}

  std::vector<std::shared_ptr<yacl::link::Context>> lctxs_;
  uint32_t thread_num_;
};

// Test ring extension from smaller bit width to larger bit width
TEST_F(RingExtensionTest, BasicRingExtensionTest) {
  const uint32_t len = 10;
  const uint32_t in_bw = 24;
  const uint32_t out_bw = 40;
  const uint64_t mod_mask = in_bw == 64 ? -1 : ((1ULL << in_bw) - 1);
  const uint64_t max_value = 1ULL << (in_bw - 2);

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;
  std::vector<uint64_t> server_x(len);
  std::vector<uint64_t> client_x(len);
  std::vector<int64_t> original_values(len);

  // Generate values in range [-max_value, max_value]
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
    RingExtensionContext ring_ext_ctx(kServerId, &rot_ctx, thread_num_);
    ring_ext_ctx.RingExtension(len, server_x.data(), server_z.data(), in_bw,
                               out_bw);
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    RingExtensionContext ring_ext_ctx(kClientId, &rot_ctx, thread_num_);
    ring_ext_ctx.RingExtension(len, client_x.data(), client_z.data(), in_bw,
                               out_bw);
  });

  server_task.get();
  client_task.get();

  // Verify: signed value should be preserved
  for (uint32_t i = 0; i < len; i++) {
    int64_t expected = original_values[i];
    int64_t computed = GetSignedValue(server_z[i] + client_z[i], out_bw);
    EXPECT_EQ(computed, expected)
        << "Ring extension test failed at index " << i
        << ", original=" << original_values[i] << ", expected=" << expected
        << ", computed=" << computed;
  }
}

TEST_F(RingExtensionTest, PreservesNegativeValueWhenLowerBitsCarry) {
  const uint32_t len = 32;
  const uint32_t in_bw = 24;
  const uint32_t out_bw = 40;
  const uint64_t quarter = 1ULL << (in_bw - 2);
  const int64_t expected = -static_cast<int64_t>(quarter);

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  // Both local MSBs are zero, but the lower bits carry into the sign bit.
  // The clear value is 3 * quarter modulo 2^in_bw, i.e. -quarter.
  std::vector<uint64_t> server_x(len, quarter + 1);
  std::vector<uint64_t> client_x(len, 2 * quarter - 1);
  std::vector<uint64_t> server_z(len);
  std::vector<uint64_t> client_z(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    RingExtensionContext ring_ext_ctx(kServerId, &rot_ctx, thread_num_);
    ring_ext_ctx.RingExtensionWithOffset(len, server_x.data(), server_z.data(),
                                         in_bw, out_bw, quarter);
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    RingExtensionContext ring_ext_ctx(kClientId, &rot_ctx, thread_num_);
    ring_ext_ctx.RingExtensionWithOffset(len, client_x.data(), client_z.data(),
                                         in_bw, out_bw, quarter);
  });

  server_task.get();
  client_task.get();

  for (uint32_t i = 0; i < len; ++i) {
    EXPECT_EQ(GetSignedValue(server_z[i] + client_z[i], out_bw), expected)
        << "Ring extension failed at index " << i;
  }
}

}  // namespace capricorn::operator_

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
