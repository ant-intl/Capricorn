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

#include "src/capricorn/activation/lut_silu_spu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <vector>

#include "gtest/gtest.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"
#include "src/capricorn/activation/secure_silu.h"
#include "src/common/nn/test_support.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/test_utils.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"

#include "yacl/crypto/rand/rand.h"

namespace capricorn::customized {

using capricorn::test::FinishTestLinks;
using capricorn::test::MatrixTestParams;

double secure_silu_plaintext(double x) {
  if (x < -8.0) {
    return 0.0;
  }
  if (x >= 8.0) {
    return x;
  }
  const auto index = std::clamp(static_cast<int>(std::floor(x + 8.0)), 0, 15);
  return silu::kSlope[index] * x + silu::kIntercept[index];
}

class _SiLUTest : public ::testing::TestWithParam<MatrixTestParams> {};

TEST_P(_SiLUTest, Work) {
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

  // CapricornLutSiLU delegates to SecureSilu, which uses a 16-segment linear
  // model.
  std::vector<double> expected(double_data.size());
  std::transform(double_data.begin(), double_data.end(), expected.begin(),
                 secure_silu_plaintext);

  auto task0 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[0]);
    env->FeedShare("data", xt_data);

    auto ctx = env->spu_ctx_.get();
    auto data_2d = env->cio_->deviceGetVar("data");

    // Test SiLU on 2D input
    auto result = CapricornLutSiLU(ctx, data_2d, 1);

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

    auto result = CapricornLutSiLU(ctx, data_2d, 2);

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

INSTANTIATE_TEST_SUITE_P(_SiLU_Instances, _SiLUTest,
                         testing::Values(MatrixTestParams{8, 8},
                                         MatrixTestParams{16, 16},
                                         MatrixTestParams{32, 32}));

}  // namespace capricorn::customized

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
