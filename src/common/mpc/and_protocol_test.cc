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

#include "src/common/mpc/and_protocol.h"

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

class AndProtocolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lctxs_ = capricorn::SetupBrpcWorld("and_protocol_test", 2);
    thread_num_ = 4;
  }

  void TearDown() override {
    auto wait = [](const std::shared_ptr<yacl::link::Context>& lctx) {
      lctx->WaitLinkTaskFinish();
    };

    auto server_wait = std::async(std::launch::async, wait, lctxs_[0]);
    auto client_wait = std::async(std::launch::async, wait, lctxs_[1]);
    server_wait.get();
    client_wait.get();
  }

  std::vector<std::shared_ptr<yacl::link::Context>> lctxs_;
  uint32_t thread_num_;
};

// Test AND gate: z = x AND x (self-AND)
TEST_F(AndProtocolTest, AndSelfTest) {
  const uint32_t len = 10;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  // Generate random boolean inputs
  emp::PRG prg;
  std::vector<uint8_t> server_x(len);
  std::vector<uint8_t> client_x(len);
  prg.random_bool(reinterpret_cast<bool*>(server_x.data()), len);
  prg.random_bool(reinterpret_cast<bool*>(client_x.data()), len);

  std::vector<uint8_t> server_z(len);
  std::vector<uint8_t> client_z(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    AndContext and_ctx(kServerId, &rot_ctx, thread_num_);
    and_ctx.And(len, server_x.data(), server_z.data());
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    AndContext and_ctx(kClientId, &rot_ctx, thread_num_);
    and_ctx.And(len, client_x.data(), client_z.data());
  });

  server_task.get();
  client_task.get();

  // Verify: z = x0 AND x1
  for (uint32_t i = 0; i < len; i++) {
    uint8_t x = server_x[i] & client_x[i];
    uint8_t z = server_z[i] ^ client_z[i];
    EXPECT_EQ(z, x) << "AND self test failed at index " << i;
  }
}

// Test AND gate: z = x AND y
TEST_F(AndProtocolTest, AndTwoInputsTest) {
  const uint32_t len = 1024;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;
  std::vector<uint8_t> server_x(len);
  std::vector<uint8_t> server_y(len);
  std::vector<uint8_t> client_x(len);
  std::vector<uint8_t> client_y(len);
  prg.random_bool(reinterpret_cast<bool*>(server_x.data()), len);
  prg.random_bool(reinterpret_cast<bool*>(server_y.data()), len);
  prg.random_bool(reinterpret_cast<bool*>(client_x.data()), len);
  prg.random_bool(reinterpret_cast<bool*>(client_y.data()), len);

  std::vector<uint8_t> server_z(len);
  std::vector<uint8_t> client_z(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    AndContext and_ctx(kServerId, &rot_ctx, thread_num_);
    and_ctx.And(len, server_x.data(), server_y.data(), server_z.data());
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    AndContext and_ctx(kClientId, &rot_ctx, thread_num_);
    and_ctx.And(len, client_x.data(), client_y.data(), client_z.data());
  });

  server_task.get();
  client_task.get();

  // Verify: z = x AND y
  for (uint32_t i = 0; i < len; i++) {
    uint8_t x = server_x[i] ^ client_x[i];
    uint8_t y = server_y[i] ^ client_y[i];
    uint8_t z = server_z[i] ^ client_z[i];
    EXPECT_EQ(z, static_cast<uint8_t>(x & y))
        << "AND two inputs test failed at index " << i;
  }
}

// Test OR gate: z = x OR x (self-OR)
TEST_F(AndProtocolTest, OrSelfTest) {
  const uint32_t len = 1024;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;
  std::vector<uint8_t> server_x(len);
  std::vector<uint8_t> client_x(len);
  prg.random_bool(reinterpret_cast<bool*>(server_x.data()), len);
  prg.random_bool(reinterpret_cast<bool*>(client_x.data()), len);

  std::vector<uint8_t> server_z(len);
  std::vector<uint8_t> client_z(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    AndContext and_ctx(kServerId, &rot_ctx, thread_num_);
    and_ctx.Or(len, server_x.data(), server_z.data());
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    AndContext and_ctx(kClientId, &rot_ctx, thread_num_);
    and_ctx.Or(len, client_x.data(), client_z.data());
  });

  server_task.get();
  client_task.get();

  // Verify: z = x0 OR x1
  for (uint32_t i = 0; i < len; i++) {
    uint8_t x = server_x[i] | client_x[i];
    uint8_t z = server_z[i] ^ client_z[i];
    EXPECT_EQ(z, x) << "OR self test failed at index " << i;
  }
}

// Test OR gate: z = x OR y
TEST_F(AndProtocolTest, OrTwoInputsTest) {
  const uint32_t len = 1024;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;
  std::vector<uint8_t> server_x(len);
  std::vector<uint8_t> server_y(len);
  std::vector<uint8_t> client_x(len);
  std::vector<uint8_t> client_y(len);
  prg.random_bool(reinterpret_cast<bool*>(server_x.data()), len);
  prg.random_bool(reinterpret_cast<bool*>(server_y.data()), len);
  prg.random_bool(reinterpret_cast<bool*>(client_x.data()), len);
  prg.random_bool(reinterpret_cast<bool*>(client_y.data()), len);

  std::vector<uint8_t> server_z(len);
  std::vector<uint8_t> client_z(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    AndContext and_ctx(kServerId, &rot_ctx, thread_num_);
    and_ctx.Or(len, server_x.data(), server_y.data(), server_z.data());
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    AndContext and_ctx(kClientId, &rot_ctx, thread_num_);
    and_ctx.Or(len, client_x.data(), client_y.data(), client_z.data());
  });

  server_task.get();
  client_task.get();

  // Verify: z = x OR y
  for (uint32_t i = 0; i < len; i++) {
    uint8_t x = server_x[i] ^ client_x[i];
    uint8_t y = server_y[i] ^ client_y[i];
    uint8_t z = server_z[i] ^ client_z[i];
    EXPECT_EQ(z, static_cast<uint8_t>(x | y))
        << "OR two inputs test failed at index " << i;
  }
}

}  // namespace capricorn::operator_

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
