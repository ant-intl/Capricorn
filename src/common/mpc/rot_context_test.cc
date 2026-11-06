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

#include "src/common/mpc/rot_context.h"

#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "gtest/gtest.h"
#include "src/common/runtime/link/mock_link_io.h"

namespace capricorn::operator_ {
namespace {

class RotContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lctxs_ = capricorn::SetupBrpcWorld("rot_context_test", 2);
  }

  void TearDown() override {}

  std::vector<std::shared_ptr<yacl::link::Context>> lctxs_;
};

// Validate ROT correctness by verifying both directions:
// 1. Client -> Server: Server's recv_msg should match client's
// sender_msg[choice]
// 2. Server -> Client: Client's recv_msg should match server's
// sender_msg[choice]
void ValidateROT(RotContext& server_ctx, RotContext& client_ctx,
                 uint32_t rot_num) {
  bool validation_passed = true;
  uint32_t error_idx = 0;

  for (uint32_t i = 0; i < rot_num && validation_passed; i++) {
    // Client -> Server direction:
    // Server's recv_msg[i] should match client's sender_msg0[i] or
    // sender_msg1[i] based on server's recv_choice[i]
    uint8_t server_choice = server_ctx.GetRecvChoice()[i];
    emp::block expected_from_client;
    if (server_choice) {
      expected_from_client = client_ctx.GetSenderMsg1()[i];
    } else {
      expected_from_client = client_ctx.GetSenderMsg0()[i];
    }

    if (!emp::cmpBlock(&server_ctx.GetRecvMsg()[i], &expected_from_client, 1)) {
      validation_passed = false;
      error_idx = i;
      break;
    }

    // Server -> Client direction:
    // Client's recv_msg[i] should match server's sender_msg0[i] or
    // sender_msg1[i] based on client's recv_choice[i]
    uint8_t client_choice = client_ctx.GetRecvChoice()[i];
    emp::block expected_from_server;
    if (client_choice) {
      expected_from_server = server_ctx.GetSenderMsg1()[i];
    } else {
      expected_from_server = server_ctx.GetSenderMsg0()[i];
    }

    if (!emp::cmpBlock(&client_ctx.GetRecvMsg()[i], &expected_from_server, 1)) {
      validation_passed = false;
      error_idx = i;
      break;
    }
  }

  EXPECT_TRUE(validation_passed)
      << "ROT validation failed at index " << error_idx;
}

TEST(RotContextConstructorTest, SupportsSingleThread) {
  RotContext context(kServerId, 1, 128);
  EXPECT_EQ(context.GetRotNumber(), 128);
}

TEST(RotContextConstructorTest, RejectsSenderRotOveruse) {
  RotContext context(kServerId, 2, 128);

  EXPECT_EQ(context.ReserveSenderRot(128), 0);
  EXPECT_THROW(context.ReserveSenderRot(1), RotContextError);
}

TEST(RotContextConstructorTest, RejectsReceiverRotOveruse) {
  RotContext context(kClientId, 2, 128);

  EXPECT_EQ(context.ReserveReceiverRot(128), 0);
  EXPECT_THROW(context.ReserveReceiverRot(1), RotContextError);
}

TEST_F(RotContextTest, ValidateROT) {
  const uint32_t thread_num = 2;
  const uint32_t rot_num = 1024;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  RotContext server_ctx(kServerId, thread_num, rot_num);
  RotContext client_ctx(kClientId, thread_num, rot_num);

  auto server_task = std::async([&]() { server_ctx.Init(server_ios); });

  auto client_task = std::async([&]() { client_ctx.Init(client_ios); });

  server_task.get();
  client_task.get();

  ValidateROT(server_ctx, client_ctx, client_ctx.GetRotNumber());
}

TEST_F(RotContextTest, RefillReusesFerretAndGeneratesFreshROT) {
  const uint32_t thread_num = 2;
  const uint32_t rot_num = 1024;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  RotContext server_ctx(kServerId, thread_num, rot_num);
  RotContext client_ctx(kClientId, thread_num, rot_num);

  auto server_init = std::async([&]() { server_ctx.Init(server_ios); });
  auto client_init = std::async([&]() { client_ctx.Init(client_ios); });
  server_init.get();
  client_init.get();

  ASSERT_EQ(server_ctx.GetGenerationCount(), 1);
  ASSERT_EQ(client_ctx.GetGenerationCount(), 1);
  ValidateROT(server_ctx, client_ctx, rot_num);

  std::vector<emp::block> first_server_sender(rot_num);
  std::memcpy(first_server_sender.data(), server_ctx.GetSenderMsg0(),
              rot_num * sizeof(emp::block));

  EXPECT_EQ(server_ctx.ReserveSenderRot(rot_num / 2), 0);
  EXPECT_EQ(server_ctx.ReserveReceiverRot(rot_num / 2), 0);
  EXPECT_EQ(client_ctx.ReserveSenderRot(rot_num / 2), 0);
  EXPECT_EQ(client_ctx.ReserveReceiverRot(rot_num / 2), 0);

  auto server_refill = std::async([&]() { server_ctx.Refill(); });
  auto client_refill = std::async([&]() { client_ctx.Refill(); });
  server_refill.get();
  client_refill.get();

  EXPECT_EQ(server_ctx.GetGenerationCount(), 2);
  EXPECT_EQ(client_ctx.GetGenerationCount(), 2);
  EXPECT_EQ(server_ctx.ReserveSenderRot(1), 0);
  EXPECT_EQ(server_ctx.ReserveReceiverRot(1), 0);
  EXPECT_EQ(client_ctx.ReserveSenderRot(1), 0);
  EXPECT_EQ(client_ctx.ReserveReceiverRot(1), 0);
  ValidateROT(server_ctx, client_ctx, rot_num);

  bool batches_are_identical = true;
  for (uint32_t index = 0; index < rot_num; ++index) {
    if (!emp::cmpBlock(&first_server_sender[index],
                       &server_ctx.GetSenderMsg0()[index], 1)) {
      batches_are_identical = false;
      break;
    }
  }
  EXPECT_FALSE(batches_are_identical);
}

}  // namespace
}  // namespace capricorn::operator_

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
