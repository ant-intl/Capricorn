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

#include "src/common/mpc/b2a.h"

#include <cstdint>
#include <future>
#include <memory>
#include <random>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "gtest/gtest.h"
#include "src/common/mpc/mux.h"
#include "src/common/mpc/rot_context.h"
#include "src/common/runtime/link/mock_link_io.h"

namespace capricorn::operator_ {

class B2ATest : public ::testing::Test {
 protected:
  void SetUp() override {
    lctxs_ = capricorn::SetupBrpcWorld("b2a_test", 2);
    thread_num_ = 4;
  }

  void TearDown() override {}

  std::vector<std::shared_ptr<yacl::link::Context>> lctxs_;
  uint32_t thread_num_;
};

// Test B2A: convert Boolean share to Arithmetic share
TEST_F(B2ATest, B2ABasicTest) {
  const uint32_t len = 16;
  const uint32_t bw = 40;
  const uint64_t mod_mask = (1ULL << bw) - 1;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;
  std::vector<uint8_t> server_choice(len);
  std::vector<uint8_t> client_choice(len);

  prg.random_bool(reinterpret_cast<bool*>(server_choice.data()), len);
  prg.random_bool(reinterpret_cast<bool*>(client_choice.data()), len);

  std::vector<uint64_t> server_z(len);
  std::vector<uint64_t> client_z(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    B2AContext b2a_ctx(kServerId, &rot_ctx, thread_num_);
    b2a_ctx.B2A(len, server_choice.data(), server_z.data(), bw);
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    B2AContext b2a_ctx(kClientId, &rot_ctx, thread_num_);
    b2a_ctx.B2A(len, client_choice.data(), client_z.data(), bw);
  });

  server_task.get();
  client_task.get();

  // Verify: z = choice (in arithmetic sharing)
  for (uint32_t i = 0; i < len; i++) {
    uint8_t choice = server_choice[i] ^ client_choice[i];
    uint64_t z = (server_z[i] + client_z[i]) & mod_mask;
    EXPECT_EQ(z, static_cast<uint64_t>(choice))
        << "B2A test failed at index " << i;
  }
}

// Test B2A with different bit widths
TEST_F(B2ATest, B2ADifferentBitWidthsTest) {
  const uint32_t len = 16;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  std::vector<uint32_t> bit_widths = {16, 32, 40, 48, 64};

  for (uint32_t bw : bit_widths) {
    uint64_t mod_mask = (bw == 64) ? -1ULL : (1ULL << bw) - 1;

    emp::PRG prg;
    std::vector<uint8_t> server_choice(len);
    std::vector<uint8_t> client_choice(len);

    prg.random_bool(reinterpret_cast<bool*>(server_choice.data()), len);
    prg.random_bool(reinterpret_cast<bool*>(client_choice.data()), len);

    std::vector<uint64_t> server_z(len);
    std::vector<uint64_t> client_z(len);

    auto server_task = std::async([&]() {
      RotContext rot_ctx(kServerId, thread_num_);
      rot_ctx.Init(server_ios);
      B2AContext b2a_ctx(kServerId, &rot_ctx, thread_num_);
      b2a_ctx.B2A(len, server_choice.data(), server_z.data(), bw);
    });

    auto client_task = std::async([&]() {
      RotContext rot_ctx(kClientId, thread_num_);
      rot_ctx.Init(client_ios);
      B2AContext b2a_ctx(kClientId, &rot_ctx, thread_num_);
      b2a_ctx.B2A(len, client_choice.data(), client_z.data(), bw);
    });

    server_task.get();
    client_task.get();

    // Verify
    for (uint32_t i = 0; i < len; i++) {
      uint8_t choice = server_choice[i] ^ client_choice[i];
      uint64_t z = (server_z[i] + client_z[i]) & mod_mask;
      EXPECT_EQ(z, static_cast<uint64_t>(choice))
          << "B2A bit width " << bw << " test failed at index " << i;
    }
  }
}

}  // namespace capricorn::operator_

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
