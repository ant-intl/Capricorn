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

#include "src/common/mpc/bole_context.h"

#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "gtest/gtest.h"
#include "src/common/mpc/elementwise.h"
#include "src/common/mpc/rot_context.h"
#include "src/common/runtime/link/mock_link_io.h"

namespace capricorn::operator_ {

class BoleContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    lctxs_ = capricorn::SetupBrpcWorld("bole_context_test", 2);
  }

  void TearDown() override {}

  std::vector<std::shared_ptr<yacl::link::Context>> lctxs_;
};

// Test: (server_x + client_x) * (server_y + client_y) == server_z + client_z
// (mod 2^bw) Note: BOLE allows off-by-one error due to the nature of the
// protocol
TEST_F(BoleContextTest, BOLECorrectness) {
  uint32_t thread_num = 16;
  const uint32_t n = 1ULL << 10;
  const uint32_t bw = 20;
  const uint64_t MOD_MASK = (bw == 64) ? -1ULL : (1ULL << bw) - 1;
  const uint64_t INPUT_MASK = (1ULL << 18) - 1;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, 1);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG server_prg;
  emp::PRG client_prg;

  std::vector<uint64_t> server_x(n);
  std::vector<uint64_t> server_y(n);
  std::vector<uint64_t> server_z(n);
  std::vector<uint64_t> client_x(n);
  std::vector<uint64_t> client_y(n);
  std::vector<uint64_t> client_z(n);

  server_prg.random_data(server_x.data(), n * sizeof(uint64_t));
  server_prg.random_data(server_y.data(), n * sizeof(uint64_t));
  ElementwiseAndInplace(n, server_x.data(), INPUT_MASK);
  ElementwiseAndInplace(n, server_y.data(), INPUT_MASK);

  client_prg.random_data(client_x.data(), n * sizeof(uint64_t));
  client_prg.random_data(client_y.data(), n * sizeof(uint64_t));
  ElementwiseAndInplace(n, client_x.data(), INPUT_MASK);
  ElementwiseAndInplace(n, client_y.data(), INPUT_MASK);

  auto server_task = std::async([&]() {
    auto rot_context = std::make_shared<RotContext>(kServerId, 1, 128);
    rot_context->Init(server_ios);
    BoleContext server_ctx(kServerId, thread_num, rot_context.get(),
                           server_ios[0].get());
    server_ctx.BOLE(n, bw, server_x.data(), server_y.data(), server_z.data());
  });

  auto client_task = std::async([&]() {
    auto rot_context = std::make_shared<RotContext>(kClientId, 1, 128);
    rot_context->Init(client_ios);
    BoleContext client_ctx(kClientId, thread_num, rot_context.get(),
                           client_ios[0].get());
    client_ctx.BOLE(n, bw, client_x.data(), client_y.data(), client_z.data());
  });

  server_task.get();
  client_task.get();

  bool flag = true;
  uint32_t error_idx = 0;
  for (uint32_t i = 0; i < n; i++) {
    uint64_t real_res =
        ((server_x[i] + client_x[i]) * (server_y[i] + client_y[i])) & MOD_MASK;
    uint64_t cal_res = (server_z[i] + client_z[i]) & MOD_MASK;

    bool cur_flag = false;
    if (real_res == cal_res) {
      cur_flag = true;
    }
    if (real_res == ((cal_res - 1) & MOD_MASK)) {
      cur_flag = true;
    }

    flag &= cur_flag;
    if (!flag) {
      error_idx = i;
      std::cout << "error at " << i << std::endl;
      std::cout << "server x = " << server_x[i] << std::endl;
      std::cout << "server y = " << server_y[i] << std::endl;
      std::cout << "client x = " << client_x[i] << std::endl;
      std::cout << "client y = " << client_y[i] << std::endl;
      std::cout << "real_res = " << real_res << std::endl;
      std::cout << "server z = " << server_z[i] << std::endl;
      std::cout << "client z = " << client_z[i] << std::endl;
      std::cout << "cal res = " << cal_res << std::endl;
      std::cout << "real - cal = " << ((real_res - cal_res) & MOD_MASK)
                << std::endl;
      break;
    }
  }

  std::cout << "BOLE test, N = " << n << ", bw = " << bw
            << ", check result = " << std::boolalpha << flag << std::endl;

  EXPECT_TRUE(flag) << "BOLE validation failed at index " << error_idx
                    << ", bw=" << bw;
}

TEST_F(BoleContextTest, BasicBOLECorrectness) {
  const uint32_t n = 1;
  const uint32_t bw = 40;
  const uint64_t MOD_MASK = (bw == 64) ? -1ULL : (1ULL << bw) - 1;
  const uint64_t INPUT_MASK = (1ULL << 40) - 1;

  auto io_pair = link::SpawnLinkIOPair(lctxs_, 1);
  auto& server_ios = io_pair.first;
  auto& client_ios = io_pair.second;

  emp::PRG prg;
  std::vector<uint64_t> server_x(n);
  std::vector<uint64_t> server_y(n);
  std::vector<uint64_t> server_z(n);
  std::vector<uint64_t> client_x(n);
  std::vector<uint64_t> client_y(n);
  std::vector<uint64_t> client_z(n);

  uint64_t origin_x = 1048576;
  uint64_t origin_y = 6287;

  prg.random_data(client_x.data(), n * sizeof(uint64_t));
  client_x[0] &= INPUT_MASK;
  server_x[0] = (origin_x - client_x[0]) & INPUT_MASK;

  prg.random_data(client_y.data(), n * sizeof(uint64_t));
  client_y[0] &= INPUT_MASK;
  server_y[0] = (origin_y - client_y[0]) & INPUT_MASK;

  // BoleContext server_ctx;
  // BoleContext client_ctx;

  auto server_task = std::async([&]() {
    auto rot_context = std::make_shared<RotContext>(kServerId, 1, 128);
    rot_context->Init(server_ios);
    BoleContext server_ctx(kServerId, 1, rot_context.get(),
                           server_ios[0].get());
    server_ctx.BOLE(n, bw, server_x.data(), server_y.data(), server_z.data());
    // server_ctx.Init(kServerId, server_ios[0].get());
    // server_ctx.BOLE(n, bw, server_x.data(), server_y.data(),
    // server_z.data());
  });

  auto client_task = std::async([&]() {
    // client_ctx.Init(kClientId, client_ios[0].get());
    // client_ctx.BOLE(n, bw, client_x.data(), client_y.data(),
    // client_z.data());
    auto rot_context = std::make_shared<RotContext>(kClientId, 1, 128);
    rot_context->Init(client_ios);
    BoleContext client_ctx(kClientId, 1, rot_context.get(),
                           client_ios[0].get());
    client_ctx.BOLE(n, bw, client_x.data(), client_y.data(), client_z.data());
  });

  server_task.get();
  client_task.get();

  bool flag = true;
  uint32_t error_idx = 0;
  for (uint32_t i = 0; i < n; i++) {
    uint64_t real_res =
        ((server_x[i] + client_x[i]) * (server_y[i] + client_y[i])) & MOD_MASK;
    uint64_t cal_res = (server_z[i] + client_z[i]) & MOD_MASK;
    uint64_t x = origin_x * origin_y;
    std::cout << "real_res = " << real_res << ", cal_res = " << cal_res
              << ", x = " << x << std::endl;

    bool cur_flag = false;
    if (real_res == cal_res) {
      cur_flag = true;
    }
    if (real_res == ((cal_res - 1) & MOD_MASK)) {
      cur_flag = true;
    }

    flag &= cur_flag;
    if (!flag) {
      error_idx = i;
      std::cout << "error at " << i << std::endl;
      std::cout << "server x = " << server_x[i] << std::endl;
      std::cout << "server y = " << server_y[i] << std::endl;
      std::cout << "client x = " << client_x[i] << std::endl;
      std::cout << "client y = " << client_y[i] << std::endl;
      std::cout << "real_res = " << real_res << std::endl;
      std::cout << "server z = " << server_z[i] << std::endl;
      std::cout << "client z = " << client_z[i] << std::endl;
      std::cout << "cal res = " << cal_res << std::endl;
      std::cout << "real - cal = " << ((real_res - cal_res) & MOD_MASK)
                << std::endl;
      break;
    }
  }

  std::cout << "BOLE test, N = " << n << ", bw = " << bw
            << ", check result = " << std::boolalpha << flag << std::endl;

  EXPECT_TRUE(flag) << "BOLE validation failed at index " << error_idx
                    << ", bw=" << bw;
}

}  // namespace capricorn::operator_

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
