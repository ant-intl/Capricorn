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

#include "src/common/mpc/compare.h"

#include <cstdint>
#include <future>
#include <memory>
#include <random>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "gtest/gtest.h"
#include "src/common/mpc/and_protocol.h"
#include "src/common/mpc/elementwise.h"
#include "src/common/mpc/rot_context.h"
#include "src/common/runtime/link/mock_link_io.h"

// Forward declaration for friend class access
class CompareTest_SubBlockCompareTest_Test;

namespace capricorn::operator_ {

class CompareTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lctxs_ = capricorn::SetupBrpcWorld("compare_test", 2);
    thread_num_ = 2;
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

// Test MSB extraction
TEST_F(CompareTest, MSBTest) {
  const uint32_t len = 10;
  const uint32_t bw = 40;
  const uint64_t mod_mask = (1ULL << bw) - 1;
  const uint64_t msb_mask = 1ULL << (bw - 1);

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;
  std::vector<uint64_t> server_x(len);
  std::vector<uint64_t> client_x(len);

  prg.random_data(server_x.data(), len * sizeof(uint64_t));
  prg.random_data(client_x.data(), len * sizeof(uint64_t));
  ElementwiseAndInplace(len, server_x.data(), mod_mask);
  ElementwiseAndInplace(len, client_x.data(), mod_mask);

  std::vector<uint8_t> server_msb(len);
  std::vector<uint8_t> client_msb(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    CompareContext compare_ctx(kServerId, &rot_ctx, thread_num_);
    compare_ctx.MSB(len, server_x.data(), server_msb.data(), bw);
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    CompareContext compare_ctx(kClientId, &rot_ctx, thread_num_);
    compare_ctx.MSB(len, client_x.data(), client_msb.data(), bw);
  });

  server_task.get();
  client_task.get();

  // Verify: msb is the most significant bit of (server_x + client_x)
  for (uint32_t i = 0; i < len; i++) {
    uint64_t x = (server_x[i] + client_x[i]) & mod_mask;
    uint8_t expected_msb = (x & msb_mask) ? 1 : 0;
    uint8_t computed_msb = server_msb[i] ^ client_msb[i];
    EXPECT_EQ(computed_msb, expected_msb) << "MSB test failed at index " << i;
  }
}

// Test Wrap (carry) detection
TEST_F(CompareTest, WrapTest) {
  const uint32_t len = 10;
  const uint32_t bw = 40;
  const uint64_t mod_mask = (1ULL << bw) - 1;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;
  std::vector<uint64_t> server_x(len);
  std::vector<uint64_t> client_x(len);

  prg.random_data(server_x.data(), len * sizeof(uint64_t));
  prg.random_data(client_x.data(), len * sizeof(uint64_t));
  ElementwiseAndInplace(len, server_x.data(), mod_mask);
  ElementwiseAndInplace(len, client_x.data(), mod_mask);

  std::vector<uint8_t> server_wrap(len);
  std::vector<uint8_t> client_wrap(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    CompareContext compare_ctx(kServerId, &rot_ctx, thread_num_);
    compare_ctx.Wrap(len, server_x.data(), server_wrap.data(), bw);
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    CompareContext compare_ctx(kClientId, &rot_ctx, thread_num_);
    compare_ctx.Wrap(len, client_x.data(), client_wrap.data(), bw);
  });

  server_task.get();
  client_task.get();

  // Verify: wrap is 1 if (server_x + client_x) >= 2^bw
  for (uint32_t i = 0; i < len; i++) {
    bool expected_wrap = (server_x[i] + client_x[i]) >= (1ULL << bw);
    uint8_t computed_wrap = server_wrap[i] ^ client_wrap[i];
    EXPECT_EQ(computed_wrap, static_cast<uint8_t>(expected_wrap))
        << "Wrap test failed at index " << i;
  }
}

// Test comparison with public value
TEST_F(CompareTest, CompareWithPublicValueTest) {
  const uint32_t len = 10;
  const uint32_t bw = 40;
  const uint64_t mod_mask = (1ULL << bw) - 1;
  const int64_t public_value = -4 * ((uint64_t)1 << 4);
  const uint64_t max_value = 8 * (1ULL << 4);

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

  std::vector<uint8_t> server_cmp(len);
  std::vector<uint8_t> client_cmp(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    CompareContext compare_ctx(kServerId, &rot_ctx, thread_num_);
    compare_ctx.Compare(len, server_x.data(),
                        static_cast<uint64_t>(public_value), server_cmp.data(),
                        bw, true, false);  // less than
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    CompareContext compare_ctx(kClientId, &rot_ctx, thread_num_);
    compare_ctx.Compare(len, client_x.data(),
                        static_cast<uint64_t>(public_value), client_cmp.data(),
                        bw, true, false);  // less than
  });

  server_task.get();
  client_task.get();

  // Verify: compare result
  for (uint32_t i = 0; i < len; i++) {
    bool expected = original_values[i] < public_value;
    uint8_t computed = server_cmp[i] ^ client_cmp[i];
    EXPECT_EQ(computed, static_cast<uint8_t>(expected))
        << "Compare with public value test failed at index " << i;
  }
}

// Test unsigned comparison
TEST_F(CompareTest, UnsignedCompareTest) {
  const uint32_t len = 10;
  const uint32_t bw = 40;
  const uint64_t mod_mask = (1ULL << bw) - 1;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;
  std::vector<uint64_t> server_x(len);
  std::vector<uint64_t> client_x(len);

  prg.random_data(server_x.data(), len * sizeof(uint64_t));
  prg.random_data(client_x.data(), len * sizeof(uint64_t));
  ElementwiseAndInplace(len, server_x.data(), mod_mask);
  ElementwiseAndInplace(len, client_x.data(), mod_mask);

  std::vector<uint8_t> server_cmp(len);
  std::vector<uint8_t> client_cmp(len);

  auto server_task = std::async([&]() {
    RotContext rot_ctx(kServerId, thread_num_);
    rot_ctx.Init(server_ios);
    CompareContext compare_ctx(kServerId, &rot_ctx, thread_num_);
    compare_ctx.UnsignedCompare(len, server_x.data(), server_cmp.data(), bw,
                                false);  // greater than
  });

  auto client_task = std::async([&]() {
    RotContext rot_ctx(kClientId, thread_num_);
    rot_ctx.Init(client_ios);
    CompareContext compare_ctx(kClientId, &rot_ctx, thread_num_);
    compare_ctx.UnsignedCompare(len, client_x.data(), client_cmp.data(), bw,
                                false);  // greater than
  });

  server_task.get();
  client_task.get();

  // Verify: server_x > client_x
  for (uint32_t i = 0; i < len; i++) {
    bool expected = server_x[i] > client_x[i];
    uint8_t computed = server_cmp[i] ^ client_cmp[i];
    EXPECT_EQ(computed, static_cast<uint8_t>(expected))
        << "Unsigned compare test failed at index " << i
        << " server_x=" << server_x[i] << " client_x=" << client_x[i];
  }
}

// Test SubBlockCompare (mill and equal test)
TEST_F(CompareTest, SubBlockCompareTest) {
  const uint32_t len = 10;
  std::vector<uint32_t> bit_widths = {4};

  for (uint32_t input_bw : bit_widths) {
    ASSERT_TRUE(1 <= input_bw && input_bw <= 8);
    const uint8_t index_mask = (input_bw == 8)
                                   ? static_cast<uint8_t>(-1)
                                   : static_cast<uint8_t>((1u << input_bw) - 1);

    auto io_pair = link::SpawnLinkIOPair(lctxs_, thread_num_);
    auto& server_ios = io_pair.first;
    auto& client_ios = io_pair.second;

    emp::PRG prg;
    std::vector<uint8_t> server_x(len);
    std::vector<uint8_t> client_x(len);

    prg.random_data(server_x.data(), len);
    prg.random_data(client_x.data(), len);
    ElementwiseAndInplace(len, server_x.data(), index_mask);
    ElementwiseAndInplace(len, client_x.data(), index_mask);

    std::vector<uint8_t> server_mill_res(len);
    std::vector<uint8_t> server_equal_res(len);
    std::vector<uint8_t> client_mill_res(len);
    std::vector<uint8_t> client_equal_res(len);

    auto server_task = std::async([&]() {
      RotContext rot_ctx(kServerId, thread_num_);
      rot_ctx.Init(server_ios);
      SPDLOG_INFO("Start SubBlockCompareTest");
      CompareContext compare_ctx(kServerId, &rot_ctx, thread_num_);
      compare_ctx.SubBlockCompare(len, server_x.data(), server_mill_res.data(),
                                  server_equal_res.data(), input_bw);
      SPDLOG_INFO("End SubBlockCompareTest");
    });

    auto client_task = std::async([&]() {
      RotContext rot_ctx(kClientId, thread_num_);
      rot_ctx.Init(client_ios);
      CompareContext compare_ctx(kClientId, &rot_ctx, thread_num_);
      compare_ctx.SubBlockCompare(len, client_x.data(), client_mill_res.data(),
                                  client_equal_res.data(), input_bw);
    });

    server_task.get();
    client_task.get();

    // Verify results
    for (uint32_t i = 0; i < len; i++) {
      // mill_res: server_x > client_x (greater than comparison)
      bool output_mill_res = server_mill_res[i] ^ client_mill_res[i];
      bool real_mill_res = (server_x[i] > client_x[i]);

      // equal_res: server_x == client_x (equality comparison)
      bool output_equal_res = server_equal_res[i] ^ client_equal_res[i];
      bool real_equal_res = (server_x[i] == client_x[i]);

      EXPECT_EQ(output_mill_res, real_mill_res)
          << "Mill test error at index " << i << " with bw=" << input_bw
          << " server_x=" << static_cast<uint32_t>(server_x[i])
          << " client_x=" << static_cast<uint32_t>(client_x[i])
          << " output=" << output_mill_res << " expected=" << real_mill_res;

      EXPECT_EQ(output_equal_res, real_equal_res)
          << "Equal test error at index " << i << " with bw=" << input_bw
          << " server_x=" << static_cast<uint32_t>(server_x[i])
          << " client_x=" << static_cast<uint32_t>(client_x[i])
          << " output=" << output_equal_res << " expected=" << real_equal_res;
    }
  }
}

}  // namespace capricorn::operator_

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
