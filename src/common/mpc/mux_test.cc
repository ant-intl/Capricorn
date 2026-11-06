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

#include "src/common/mpc/mux.h"

#include <cstdint>
#include <future>
#include <memory>
#include <random>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "gtest/gtest.h"
#include "src/common/mpc/rot_context.h"
#include "src/common/runtime/link/mock_link_io.h"

namespace capricorn::operator_ {

class MuxTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lctxs_ = capricorn::SetupBrpcWorld("mux_test", 2);
    thread_num_ = 4;
  }

  void TearDown() override {}

  std::vector<std::shared_ptr<yacl::link::Context>> lctxs_;
  uint32_t thread_num_;
};

// Test MUX: z = choice ? value : 0
TEST_F(MuxTest, MuxWithZeroTest) {
  const uint32_t len = 1024;
  const uint32_t bw = 40;
  const uint64_t mod_mask = (1ULL << bw) - 1;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;
  std::vector<uint8_t> server_choice(len);
  std::vector<uint8_t> client_choice(len);
  std::vector<uint64_t> server_value(len);
  std::vector<uint64_t> client_value(len);

  prg.random_bool(reinterpret_cast<bool*>(server_choice.data()), len);
  prg.random_bool(reinterpret_cast<bool*>(client_choice.data()), len);
  prg.random_data(server_value.data(), len * sizeof(uint64_t));
  prg.random_data(client_value.data(), len * sizeof(uint64_t));

  for (uint32_t i = 0; i < len; i++) {
    server_value[i] &= mod_mask;
    client_value[i] &= mod_mask;
  }

  std::vector<uint64_t> server_z(len);
  std::vector<uint64_t> client_z(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    MuxContext mux_ctx(kServerId, &rot_ctx, thread_num_);
    mux_ctx.Mux(len, server_choice.data(), server_value.data(), server_z.data(),
                bw);
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    MuxContext mux_ctx(kClientId, &rot_ctx, thread_num_);
    mux_ctx.Mux(len, client_choice.data(), client_value.data(), client_z.data(),
                bw);
  });

  server_task.get();
  client_task.get();

  // Verify: z = choice ? value : 0
  for (uint32_t i = 0; i < len; i++) {
    uint8_t choice = server_choice[i] ^ client_choice[i];
    uint64_t value = (server_value[i] + client_value[i]) & mod_mask;
    uint64_t z = (server_z[i] + client_z[i]) & mod_mask;
    uint64_t expected = choice ? value : 0;
    EXPECT_EQ(z, expected) << "MUX with zero test failed at index " << i;
  }
}

// Test MUX: z = choice ? y : x
TEST_F(MuxTest, MuxTwoValuesTest) {
  const uint32_t len = 1024;
  const uint32_t bw = 40;
  const uint64_t mod_mask = (1ULL << bw) - 1;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;
  std::vector<uint8_t> server_choice(len);
  std::vector<uint8_t> client_choice(len);
  std::vector<uint64_t> server_x(len);
  std::vector<uint64_t> server_y(len);
  std::vector<uint64_t> client_x(len);
  std::vector<uint64_t> client_y(len);

  prg.random_bool(reinterpret_cast<bool*>(server_choice.data()), len);
  prg.random_bool(reinterpret_cast<bool*>(client_choice.data()), len);
  prg.random_data(server_x.data(), len * sizeof(uint64_t));
  prg.random_data(server_y.data(), len * sizeof(uint64_t));
  prg.random_data(client_x.data(), len * sizeof(uint64_t));
  prg.random_data(client_y.data(), len * sizeof(uint64_t));

  for (uint32_t i = 0; i < len; i++) {
    server_x[i] &= mod_mask;
    server_y[i] &= mod_mask;
    client_x[i] &= mod_mask;
    client_y[i] &= mod_mask;
  }

  std::vector<uint64_t> server_z(len);
  std::vector<uint64_t> client_z(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    MuxContext mux_ctx(kServerId, &rot_ctx, thread_num_);
    mux_ctx.Mux(len, server_choice.data(), server_x.data(), server_y.data(),
                server_z.data(), bw);
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    MuxContext mux_ctx(kClientId, &rot_ctx, thread_num_);
    mux_ctx.Mux(len, client_choice.data(), client_x.data(), client_y.data(),
                client_z.data(), bw);
  });

  server_task.get();
  client_task.get();

  // Verify: z = choice ? y : x
  for (uint32_t i = 0; i < len; i++) {
    uint8_t choice = server_choice[i] ^ client_choice[i];
    uint64_t x = (server_x[i] + client_x[i]) & mod_mask;
    uint64_t y = (server_y[i] + client_y[i]) & mod_mask;
    uint64_t z = (server_z[i] + client_z[i]) & mod_mask;
    uint64_t expected = choice ? y : x;
    EXPECT_EQ(z, expected) << "MUX two values test failed at index " << i;
  }
}

}  // namespace capricorn::operator_

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
