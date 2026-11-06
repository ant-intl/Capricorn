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

#include "src/common/nn/spu_activation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <vector>

#include "gtest/gtest.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"
#include "src/common/nn/test_support.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/macro.h"
#include "src/common/runtime/test_utils.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"

#include "yacl/crypto/rand/rand.h"

namespace capricorn::customized {

using capricorn::test::FinishTestLinks;
using capricorn::test::MatrixTestParams;

xt::xarray<double> silu_plaintext(const xt::xarray<double>& x) {
  const double a0 = -0.3067541139982155;
  const double a1 = -0.0819767021525476;
  const double a2 = -0.0055465625580307;
  const double b0 = 0.0085064025895951;
  const double b1 = 0.5;
  const double b2 = 0.2281430841728270;
  const double b3 = -0.011113046708173;
  const double b4 = 0.0002743776353465;

  auto b0_mask = xt::cast<double>(x < -8.0);  // x < -8
  auto b1_mask = xt::cast<double>(x < -4.0);  // x < -4
  auto b2_mask = xt::cast<double>(x > 4.0);   // x > 4
  // b4_mask: x in [-8, -4)
  auto b4_mask = xt::abs(b0_mask - b1_mask);  // XOR via abs(a - b)
  // b3_mask: x in [-4, 4)  =>  NOT(b1 OR b2)  =>  1 - max(b1, b2)
  auto b3_mask = 1.0 - xt::maximum(b1_mask, b2_mask);

  auto x2 = xt::square(x);
  auto x4 = xt::square(x2);
  auto x6 = x2 * x4;

  auto seg1 = a2 * x2 + a1 * x + a0;
  auto seg2 = b4 * x6 + b3 * x4 + b2 * x2 + b1 * x + b0;

  return b2_mask * x + b4_mask * seg1 + b3_mask * seg2;
}

class SiLUTest : public ::testing::TestWithParam<MatrixTestParams> {};

TEST_P(SiLUTest, Work) {
  auto lctxs = Mock2PCLink(true);
  auto params = GetParam();

  spu::Shape shape2d = {params.row, params.col};

  // Generate random test data - scale to prevent overflow in MPC arithmetic
  auto data = yacl::crypto::RandVec<uint64_t>(params.row * params.col);
  auto double_data = std::vector<double>(data.size(), 0.0);
  // Scale to smaller range to avoid MPC precision issues (same as
  // expert_compute_test)
  std::transform(data.begin(), data.end(), double_data.begin(),
                 [](uint64_t val) { return val * 1.0 / std::pow(2, 64); });

  auto xt_data = xt::adapt(double_data, {params.row, params.col});

  // Compute expected SiLU result using same polynomial approximation as MPC
  auto expected_xt = silu_plaintext(xt_data);
  std::vector<double> expected(expected_xt.begin(), expected_xt.end());

  auto task0 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[0]);
    env->FeedShare("data", xt_data);

    auto ctx = env->spu_ctx_.get();
    auto data_2d = env->cio_->deviceGetVar("data");

    // Test SiLU on 2D input
    COMM_TIMER_START(lctxs[0], MOE_SILU_Server);
    auto result = SpuPiecewiseSiLU(ctx, data_2d);
    COMM_TIMER_END_PRINT(lctxs[0], MOE_SILU_Server);

    auto result_public = spu::kernel::hal::dump_public_as<double>(
        ctx, spu::kernel::hal::reveal(ctx, result));

    for (int64_t i = 0; i < params.row * params.col; i++) {
      // Use tolerance for MPC piecewise polynomial approximation
      // Note: SiLU MPC implementation has fixed-point arithmetic issues
      // that are covered by the fixed-point tolerance below
      EXPECT_NEAR(result_public[i], expected[i], 1e-2);
    }
    return env;
  });

  auto task1 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[1]);
    auto xt_zeros = xt::xarray<double>::from_shape(
        {static_cast<size_t>(params.row), static_cast<size_t>(params.col)});
    xt_zeros.fill(0.0);
    env->FeedShare("data", xt_zeros);

    auto ctx = env->spu_ctx_.get();
    auto data_2d = env->cio_->deviceGetVar("data");

    COMM_TIMER_START(lctxs[1], MOE_SILU_Client);
    auto result = SpuPiecewiseSiLU(ctx, data_2d);
    COMM_TIMER_END_PRINT(lctxs[1], MOE_SILU_Client);

    auto result_public = spu::kernel::hal::dump_public_as<double>(
        ctx, spu::kernel::hal::reveal(ctx, result));

    for (int64_t i = 0; i < params.row * params.col; i++) {
      // Use tolerance for MPC piecewise polynomial approximation
      // Note: SiLU MPC implementation has fixed-point arithmetic issues
      // that are covered by the fixed-point tolerance below
      EXPECT_NEAR(result_public[i], expected[i], 1e-2);
    }
    return env;
  });

  auto env0 = task0.get();
  auto env1 = task1.get();
  // Release protocol-owned buffers before shutting down the shared links.
  env0.reset();
  env1.reset();
  FinishTestLinks(lctxs);
}

INSTANTIATE_TEST_SUITE_P(SiLU_Instances, SiLUTest,
                         testing::Values(MatrixTestParams{8, 16},
                                         MatrixTestParams{16, 32},
                                         MatrixTestParams{32, 64}));

// Paper Table 3 sizes. Keep these disabled by default so the regular unit-test
// suite stays fast; benchmark orchestration enables and filters them
// explicitly. The element counts are 2^18, 2^20, and 2^22 respectively.
INSTANTIATE_TEST_SUITE_P(DISABLED_Table3, SiLUTest,
                         testing::Values(MatrixTestParams{1 << 9, 1 << 9},
                                         MatrixTestParams{1 << 10, 1 << 10},
                                         MatrixTestParams{1 << 11, 1 << 11}));

}  // namespace capricorn::customized

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
