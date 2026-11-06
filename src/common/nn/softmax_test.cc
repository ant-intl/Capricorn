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

#include "src/common/nn/softmax.h"

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
#include "src/common/runtime/test_utils.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"

#include "yacl/crypto/rand/rand.h"

namespace capricorn::customized {

using capricorn::test::FinishTestLinks;
using capricorn::test::MatrixTestParams;

class SoftmaxTest : public ::testing::TestWithParam<MatrixTestParams> {};

TEST_P(SoftmaxTest, Work) {
  auto lctxs = Mock2PCLink(true);
  auto params = GetParam();

  spu::Shape shape2d = {params.row, params.col};

  // Generate random test data - use smaller range for MPC fixed-point
  // arithmetic
  auto data = yacl::crypto::RandVec<uint64_t>(params.row * params.col);
  auto double_data = std::vector<double>(data.size(), 0.0);
  // Scale to smaller range to avoid MPC precision issues
  std::transform(data.begin(), data.end(), double_data.begin(),
                 [](uint64_t val) { return val * 1.0 / std::pow(2, 60); });

  auto xt_data = xt::adapt(double_data, {params.row, params.col});

  // Compute expected softmax result along axis 1
  std::vector<double> expected(params.row * params.col);
  for (int64_t i = 0; i < params.row; i++) {
    // Find max for numerical stability
    double max_val = double_data[i * params.col];
    for (int64_t j = 1; j < params.col; ++j) {
      max_val = std::max(max_val, double_data[i * params.col + j]);
    }

    // Compute exp(x - max) and sum
    double sum_exp = 0.0;
    std::vector<double> exp_vals(params.col);
    for (int64_t j = 0; j < params.col; ++j) {
      exp_vals[j] = std::exp(double_data[i * params.col + j] - max_val);
      sum_exp += exp_vals[j];
    }

    // Normalize
    for (int64_t j = 0; j < params.col; ++j) {
      expected[i * params.col + j] = exp_vals[j] / sum_exp;
    }
  }

  auto task0 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[0]);
    env->FeedShare("data", xt_data);

    auto ctx = env->spu_ctx_.get();
    auto data_2d = env->cio_->deviceGetVar("data");

    // Test Softmax along axis 1
    auto result = Softmax_2d(ctx, data_2d, 1);

    auto result_public = spu::kernel::hal::dump_public_as<double>(
        ctx, spu::kernel::hal::reveal(ctx, result));

    for (int64_t i = 0; i < params.row * params.col; i++) {
      // Use larger tolerance due to MPC precision limitations
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

    auto result = Softmax_2d(ctx, data_2d, 1);

    auto result_public = spu::kernel::hal::dump_public_as<double>(
        ctx, spu::kernel::hal::reveal(ctx, result));

    for (int64_t i = 0; i < params.row * params.col; i++) {
      // Use larger tolerance due to MPC precision limitations
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

INSTANTIATE_TEST_SUITE_P(Softmax_Instances, SoftmaxTest,
                         testing::Values(MatrixTestParams{8, 8},
                                         MatrixTestParams{16, 16},
                                         MatrixTestParams{32, 32}));

}  // namespace capricorn::customized

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
