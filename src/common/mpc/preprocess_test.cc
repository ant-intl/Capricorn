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

#include "src/common/mpc/preprocess.h"

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

class PreprocessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lctxs_ = capricorn::SetupBrpcWorld("preprocess_test", 2);
  }

  void TearDown() override {}

  std::vector<std::shared_ptr<yacl::link::Context>> lctxs_;
};

// Validate that AES keys are synchronized between server and client
void ValidateAesKeySync(const emp::block& server_key,
                        const emp::block& client_key) {
  EXPECT_TRUE(emp::cmpBlock(&server_key, &client_key, 1))
      << "AES keys do not match between server and client";
}

// Helper function to generate random data
void GenerateRandomData(uint64_t* data, uint32_t len, uint64_t mod_mask,
                        std::mt19937_64& rng) {
  std::uniform_int_distribution<uint64_t> dist(0, mod_mask);
  for (uint32_t i = 0; i < len; i++) {
    data[i] = dist(rng);
  }
}

// Test basic Preprocess functionality with minimal thread count
TEST_F(PreprocessTest, PreprocessBasicCorrectness) {
  const uint32_t thread_num = 2;
  const uint32_t rot_num = 1024;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  PreprocessResult server_result;
  PreprocessResult client_result;

  auto server_task = std::async([&]() {
    server_result = Preprocess(server_ios, kServerId, thread_num, rot_num);
  });

  auto client_task = std::async([&]() {
    client_result = Preprocess(client_ios, kClientId, thread_num, rot_num);
  });

  server_task.get();
  client_task.get();

  // Verify that contexts are initialized
  ASSERT_NE(server_result.rot_context, nullptr);
  ASSERT_NE(server_result.bole_context, nullptr);
  ASSERT_NE(client_result.rot_context, nullptr);
  ASSERT_NE(client_result.bole_context, nullptr);

  // Verify party IDs
  EXPECT_EQ(server_result.rot_context->GetPartyId(), kServerId);
  EXPECT_EQ(client_result.rot_context->GetPartyId(), kClientId);

}  // namespace capricorn::operator_
}  // namespace capricorn::operator_

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
