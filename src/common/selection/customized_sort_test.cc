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

#include "src/common/selection/customized_sort.h"

#include <algorithm>
#include <cstdint>
#include <future>
#include <numeric>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/rank.h"
#include "spdlog/spdlog.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/test_utils.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"

#include "yacl/crypto/rand/rand.h"

namespace capricorn::customized {

struct Sort2dTestParams {
  int64_t row;
  int64_t width;
};

class Sort2dTest : public ::testing::TestWithParam<Sort2dTestParams> {};

// Plaintext reference implementation of gate routing using xtensor
// Computes: W, K = Top-K(Softmax(Linear(x)), k)

TEST_P(Sort2dTest, Work) {
  auto lctxs = Mock2PCLink(false);
  auto params = GetParam();

  spu::Shape shape2d = {params.row, params.width};

  auto payload = yacl::crypto::RandVec<uint64_t>(params.row * params.width);

  auto double_key = std::vector<double>(params.row * params.width, 0.0f);
  auto double_payload = std::vector<double>(payload.size(), 0.0f);
  auto double_zeros = std::vector<double>(double_key.size(), 0.0f);

  // The runtime encodes floating-point inputs with 18 fractional bits. Two
  // independently generated doubles can therefore become equal after fixed-
  // point quantization, leaving their payload order unspecified. Use shuffled,
  // exactly representable keys so this test exercises sorting without ties.
  std::mt19937_64 prng(0xC0FFEE);
  std::vector<int64_t> row_keys(params.width);
  std::iota(row_keys.begin(), row_keys.end(), 0);
  for (int64_t i = 0; i < params.row; ++i) {
    std::shuffle(row_keys.begin(), row_keys.end(), prng);
    for (int64_t j = 0; j < params.width; ++j) {
      double_key[i * params.width + j] =
          static_cast<double>(row_keys[j] + 1) / 256.0;
    }
  }
  std::transform(payload.begin(), payload.end(), double_payload.begin(),
                 [](uint64_t val) { return val * 1.0 / std::pow(2, 60); });

  auto xt_key = xt::adapt(double_key, {params.row, params.width});
  auto xt_payload = xt::adapt(double_payload, {params.row, params.width});
  auto xt_zeros = xt::adapt(double_zeros, {params.row, params.width});

  std::vector<std::pair<double, double>> expected;

  for (int64_t i = 0; i < params.row; i++) {
    int64_t offset = i * params.width;
    auto expect = std::map<double, double, std::greater<double>>();
    for (int64_t j = 0; j < params.width; ++j) {
      int64_t loc = offset + j;
      expect[double_key[loc]] = double_payload[loc];
    }

    for (auto& it : expect) {
      expected.push_back({it.first, it.second});
    }
  }

  auto task0 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[0]);
    env->FeedShare("key", xt_key);
    // env->FeedShare("payload", xt_payload);
    env->SetPublic("payload", xt_payload);
    env->FeedShare("zeros", xt_zeros);

    auto ctx = env->spu_ctx_.get();
    auto key_2d = env->cio_->deviceGetVar("key");
    auto payload_2d = env->cio_->deviceGetVar("payload");
    auto zeros = env->cio_->deviceGetVar("zeros");

    // payload_2d = spu::kernel::hal::_xor(ctx, payload_2d, zeros)
    //                  .setDtype(payload_2d.dtype());

    spu::kernel::hlo::TopK(ctx, key_2d, 10);

    auto ret = sort_2d(ctx, key_2d, {payload_2d});

    auto sorted_key = spu::kernel::hal::dump_public_as<double>(
        ctx, spu::kernel::hal::reveal(ctx, ret[0]));
    auto sorted_payload = spu::kernel::hal::dump_public_as<double>(
        ctx, spu::kernel::hal::reveal(ctx, ret[1]));

    for (int64_t i = 0; i < params.row; i++) {
      int64_t offset = i * params.width;
      for (int64_t j = 0; j < params.width; ++j) {
        int64_t loc = offset + j;
        EXPECT_NEAR(sorted_key[loc], expected[loc].first, 1e-3);
        EXPECT_NEAR(sorted_payload[loc], expected[loc].second, 1e-3);
      }
    }
    return env;
  });
  auto task1 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[1]);
    env->FeedShare("key", xt_zeros);
    // env->FeedShare("payload", xt_zeros);
    env->SetPublic("payload", xt_payload);
    env->FeedShare("zeros", xt_zeros);

    auto ctx = env->spu_ctx_.get();
    auto key_2d = env->cio_->deviceGetVar("key");
    auto payload_2d = env->cio_->deviceGetVar("payload");
    auto zeros = env->cio_->deviceGetVar("zeros");

    // payload_2d = spu::kernel::hal::_xor(ctx, payload_2d, zeros)
    //                  .setDtype(payload_2d.dtype());

    SPDLOG_INFO("BEGIN TopK");
    spu::kernel::hlo::TopK(ctx, key_2d, 10);
    SPDLOG_INFO("END TopK");

    SPDLOG_INFO("BEGIN SORT 2D");
    auto ret = sort_2d(ctx, key_2d, {payload_2d});
    SPDLOG_INFO("END SORT 2D");

    auto sorted_key = spu::kernel::hal::dump_public_as<double>(
        ctx, spu::kernel::hal::reveal(ctx, ret[0]));
    auto sorted_payload = spu::kernel::hal::dump_public_as<double>(
        ctx, spu::kernel::hal::reveal(ctx, ret[1]));

    for (int64_t i = 0; i < params.row; i++) {
      int64_t offset = i * params.width;
      for (int64_t j = 0; j < params.width; ++j) {
        int64_t loc = offset + j;
        EXPECT_NEAR(sorted_key[loc], expected[loc].first, 1e-3);
        EXPECT_NEAR(sorted_payload[loc], expected[loc].second, 1e-3);
      }
    }
    return env;
  });

  auto env0 = task0.get();
  auto env1 = task1.get();
}

TEST(BatchedQuickSelectTopK2dTest, MatchesPlainTopKPerRow) {
  constexpr int64_t kRows = 4;
  constexpr int64_t kWidth = 12;
  constexpr int64_t k = 3;

  auto lctxs = Mock2PCLink(false);
  std::vector<double> keys(kRows * kWidth);
  std::vector<double> zeros(keys.size(), 0.0);
  for (int64_t row = 0; row < kRows; ++row) {
    for (int64_t column = 0; column < kWidth; ++column) {
      // Distinct, exactly representable fixed-point values with a different
      // ordering in every row.
      const int64_t rank = (column * 5 + row * 3) % kWidth;
      keys[row * kWidth + column] =
          static_cast<double>(row * kWidth + rank + 1) / 256.0;
    }
  }

  std::vector<std::vector<double>> expected(kRows);
  for (int64_t row = 0; row < kRows; ++row) {
    expected[row].assign(keys.begin() + row * kWidth,
                         keys.begin() + (row + 1) * kWidth);
    std::sort(expected[row].begin(), expected[row].end(),
              std::greater<double>());
    expected[row].resize(k);
  }

  auto xt_keys = xt::adapt(keys, {kRows, kWidth});
  auto xt_zeros = xt::adapt(zeros, {kRows, kWidth});

  auto run_party = [&](size_t rank) {
    auto env = std::make_unique<Environment>(lctxs[rank]);
    env->FeedShare("key", rank == 0 ? xt_keys : xt_zeros);

    auto* ctx = env->spu_ctx_.get();
    auto input = env->cio_->deviceGetVar("key");
    auto result = batched_quick_select_topk_2d(ctx, input, k);
    auto top_values = spu::kernel::hal::dump_public_as<double>(
        ctx, spu::kernel::hal::reveal(ctx, result[0]));
    auto top_indices = spu::kernel::hal::dump_public_as<int64_t>(
        ctx, spu::kernel::hal::reveal(ctx, result[1]));

    for (int64_t row = 0; row < kRows; ++row) {
      std::vector<double> actual;
      actual.reserve(k);
      for (int64_t slot = 0; slot < k; ++slot) {
        const int64_t offset = row * k + slot;
        EXPECT_GE(top_indices[offset], 0);
        EXPECT_LT(top_indices[offset], kWidth);
        if (top_indices[offset] >= 0 && top_indices[offset] < kWidth) {
          EXPECT_NEAR(top_values[offset],
                      keys[row * kWidth + top_indices[offset]], 1e-3);
        }
        actual.emplace_back(top_values[offset]);
      }
      std::sort(actual.begin(), actual.end(), std::greater<double>());
      for (int64_t slot = 0; slot < k; ++slot) {
        EXPECT_NEAR(actual[slot], expected[row][slot], 1e-3);
      }
    }
    return env;
  };

  auto task0 = std::async(std::launch::async, run_party, 0);
  auto task1 = std::async(std::launch::async, run_party, 1);
  auto env0 = task0.get();
  auto env1 = task1.get();
}

INSTANTIATE_TEST_SUITE_P(Works_Instances, Sort2dTest,
                         testing::Values(Sort2dTestParams{64, 32},
                                         Sort2dTestParams{128, 64},
                                         Sort2dTestParams{32, 64},
                                         Sort2dTestParams{64, 128}));

}  // namespace capricorn::customized

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
