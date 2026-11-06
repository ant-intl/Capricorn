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

#include "src/capricorn/expert_compute.h"

#include <algorithm>
#include <cstdint>
#include <future>

#include "gtest/gtest.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/test_utils.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xcsv.hpp"
#include "xtensor/views/xview.hpp"

#include "yacl/crypto/rand/rand.h"

namespace capricorn {

struct SiLUExpertComputeTestParams {
  size_t expert_nums;
  size_t dispatch_expert_per_token{16};
  size_t embedding_size{64};
  size_t expert_dimension{128};
};

class SiLUExpertComputeTest
    : public ::testing::TestWithParam<SiLUExpertComputeTestParams> {};

xt::xarray<double> matmul(const xt::xarray<double>& A,
                          const xt::xarray<double>& B) {
  size_t M = A.shape()[0];
  YACL_ENFORCE(A.shape()[1] == B.shape()[0]);
  size_t N = B.shape()[1];

  xt::xarray<double> C = xt::zeros<double>({M, N});

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      C(i, j) =
          xt::sum(xt::view(A, i, xt::all()) * xt::view(B, xt::all(), j))();
    }
  }
  return C;
}

// FIXME: @linhui
xt::xarray<double> _silu(const xt::xarray<double>& x) { return x; }

std::vector<double> PlaintextSiLUExpertCompute(
    const SiLUExpertComputeTestParams& param,
    const std::vector<double>& selected_embeddings_for_experts,
    const std::vector<double>& experts_weights_one,
    const std::vector<double>& experts_weights_two) {
  int64_t n = param.expert_nums;
  // int64_t k = param.active_expert_nums;
  int64_t t = param.dispatch_expert_per_token;
  int64_t d = param.embedding_size;
  int64_t ed = param.expert_dimension;

  SPDLOG_INFO("n: {}, t: {}, d: {} , ed: {}", n, t, d, ed);

  std::vector<double> tmp;

  for (int i = 0; i < n; ++i) {
    auto selected_embeddings_for_experts_slice = std::vector<double>(
        selected_embeddings_for_experts.begin() + i * t * d,
        selected_embeddings_for_experts.begin() + i * t * d + t * d);
    auto experts_weights_one_slice =
        std::vector<double>(experts_weights_one.begin() + i * d * ed,
                            experts_weights_one.begin() + i * d * ed + d * ed);
    auto xt_slice = xt::adapt(selected_embeddings_for_experts_slice, {t, d});
    auto xt_slice2 = xt::adapt(experts_weights_one_slice, {d, ed});
    // First linear transformation: x @ W1
    auto xt_slice3 = matmul(xt_slice, xt_slice2);
    // Apply SILU activation
    auto xt_slice3_activated = _silu(xt_slice3);

    std::copy(xt_slice3_activated.begin(), xt_slice3_activated.end(),
              std::back_inserter(tmp));
  }

  std::vector<double> result;
  for (int i = 0; i < n; ++i) {
    auto tmp_slice = std::vector<double>(tmp.begin() + i * t * ed,
                                         tmp.begin() + i * t * ed + t * ed);
    auto xt_slice = xt::adapt(tmp_slice, {t, ed});
    auto experts_weights_two_slice =
        std::vector<double>(experts_weights_two.begin() + i * ed * d,
                            experts_weights_two.begin() + i * ed * d + ed * d);
    auto xt_slice2 = xt::adapt(experts_weights_two_slice, {ed, d});
    // Second linear transformation: activated_hidden @ W2
    auto xt_slice3 = matmul(xt_slice, xt_slice2);

    std::copy(xt_slice3.begin(), xt_slice3.end(), std::back_inserter(result));
  }

  return result;
}

TEST(SiLUExpertComputeKernelTest, ReturnsSecretExpertTensor) {
  auto lctxs = Mock2PCLink(true);
  const SiLUExpertComputeConfig config(
      /*expert_nums=*/2, /*token_capacity_per_expert=*/8,
      /*embedding_dimension=*/4, /*expert_dimension=*/4);
  // SecureSiLU currently parallelizes a minimum protocol batch of 64 values.
  // Keep the direct kernel test at that boundary (2 * 8 * 4) so it exercises
  // the production protocol rather than an unsupported tiny batch.
  std::vector<double> selected_values(64);
  for (size_t i = 0; i < selected_values.size(); ++i) {
    selected_values[i] = static_cast<double>(i % 8) / 16.0 - 0.25;
  }
  const std::vector<double> zero_selected(64, 0.0);
  std::vector<double> w1_values(32, 0.0);
  std::vector<double> w2_values(32, 0.0);
  for (size_t expert = 0; expert < 2; ++expert) {
    for (size_t i = 0; i < 4; ++i) {
      w1_values[expert * 16 + i * 4 + i] = 0.25;
      w2_values[expert * 16 + i * 4 + i] = 0.25;
    }
  }

  auto run_party = [&](size_t rank) {
    Environment environment(lctxs[rank]);
    auto selected_data = rank == 0 ? selected_values : zero_selected;
    auto selected_tensor = xt::adapt(selected_data, {2, 8, 4});
    environment.FeedShare(config.kSelectedEmbeddingsForExperts,
                          selected_tensor);

    if (rank == 0) {
      auto w1_data = w1_values;
      auto w2_data = w2_values;
      auto w1_tensor = xt::adapt(w1_data, {2, 4, 4});
      auto w2_tensor = xt::adapt(w2_data, {2, 4, 4});
      environment.SetPrivate(config.kExpertsWeightsOne, w1_tensor);
      environment.SetPrivate(config.kExpertsWeightsTwo, w2_tensor);
    } else {
      environment.GetPrivate(config.kExpertsWeightsOne);
      environment.GetPrivate(config.kExpertsWeightsTwo);
    }

    auto output = SiLUExpertComputeKernel(
        environment.spu_ctx_.get(),
        environment.cio_->deviceGetVar(config.kSelectedEmbeddingsForExperts),
        environment.cio_->deviceGetVar(config.kExpertsWeightsOne),
        environment.cio_->deviceGetVar(config.kExpertsWeightsTwo), config);
    EXPECT_EQ(output.shape(), spu::Shape({2, 8, 4}));
    EXPECT_TRUE(output.isSecret());
  };

  auto party_zero = std::async(std::launch::async, run_party, 0);
  run_party(1);
  party_zero.get();
}

TEST(SiLUExpertComputeKernelTest, SecureSiluConsumesBothExpertBiases) {
  auto lctxs = Mock2PCLink(true);
  const SiLUExpertComputeConfig config(
      /*expert_nums=*/2, /*token_capacity_per_expert=*/8,
      /*embedding_dimension=*/4, /*expert_dimension=*/4);
  const std::vector<double> selected(64, 0.0);
  const std::vector<double> w1(32, 0.0);
  std::vector<double> w2(32, 0.0);
  for (size_t expert = 0; expert < 2; ++expert) {
    for (size_t i = 0; i < 4; ++i) {
      w2[expert * 16 + i * 4 + i] = 0.25;
    }
  }
  const std::vector<double> b1(8, 0.5);
  const std::vector<double> b2(8, 0.125);

  auto run_party = [&](size_t rank) {
    Environment environment(lctxs[rank]);
    environment.FeedShare("bias_selected", xt::adapt(selected, {2, 8, 4}));
    if (rank == 0) {
      auto w1_copy = w1;
      auto b1_copy = b1;
      auto w2_copy = w2;
      auto b2_copy = b2;
      environment.SetPrivate("bias_w1", xt::adapt(w1_copy, {2, 4, 4}));
      environment.SetPrivate("bias_b1", xt::adapt(b1_copy, {2, 4}));
      environment.SetPrivate("bias_w2", xt::adapt(w2_copy, {2, 4, 4}));
      environment.SetPrivate("bias_b2", xt::adapt(b2_copy, {2, 4}));
    } else {
      environment.GetPrivate("bias_w1");
      environment.GetPrivate("bias_b1");
      environment.GetPrivate("bias_w2");
      environment.GetPrivate("bias_b2");
    }
    auto output = SiLUExpertComputeKernelWithBias(
        environment.spu_ctx_.get(),
        environment.cio_->deviceGetVar("bias_selected"),
        environment.cio_->deviceGetVar("bias_w1"),
        environment.cio_->deviceGetVar("bias_b1"),
        environment.cio_->deviceGetVar("bias_w2"),
        environment.cio_->deviceGetVar("bias_b2"), config);
    auto revealed =
        spu::kernel::hal::reveal(environment.spu_ctx_.get(), output);
    auto values = spu::kernel::hal::dump_public_as<double>(
        environment.spu_ctx_.get(), revealed);
    return std::vector<double>(values.begin(), values.end());
  };

  auto party_zero = std::async(std::launch::async, run_party, 0);
  auto result_one = run_party(1);
  auto result_zero = party_zero.get();
  ASSERT_EQ(result_zero.size(), 64);
  ASSERT_EQ(result_zero, result_one);
  // In the [0, 1) segment, SecureSiLU uses slope 0.7345586419 and
  // intercept -0.0378005020. The expected value is 0.25 * SiLU(0.5) + b2.
  // Checking every value catches omission of either bias or an identity
  // activation, which a positive-output-only check cannot distinguish.
  constexpr double expected =
      0.25 * (0.7345586419 * 0.5 - 0.0378005020) + 0.125;
  for (double actual : result_zero) {
    EXPECT_NEAR(actual, expected, 0.002);
  }
}

TEST_P(SiLUExpertComputeTest, Work) {
  // Setup 2-party communication channels
  auto lctxs = Mock2PCLink(true);
  auto params = GetParam();

  SiLUExpertComputeConfig config(
      params.expert_nums, params.dispatch_expert_per_token,
      params.embedding_size, params.expert_dimension);

  auto selected_embeddings_for_experts = yacl::crypto::RandVec<uint64_t>(
      params.expert_nums * params.dispatch_expert_per_token *
      params.embedding_size);
  auto experts_weights_one = yacl::crypto::RandVec<uint64_t>(
      params.expert_nums * params.embedding_size * params.expert_dimension);
  auto experts_weights_two = yacl::crypto::RandVec<uint64_t>(
      params.expert_nums * params.expert_dimension * params.embedding_size);

  auto double_selected_embeddings_for_experts =
      std::vector<double>(selected_embeddings_for_experts.size(), 0);
  auto double_experts_weights_one =
      std::vector<double>(experts_weights_one.size(), 0);
  auto double_experts_weights_two =
      std::vector<double>(experts_weights_two.size(), 0);
  // Reduce value range to prevent overflow in MPC arithmetic
  std::transform(selected_embeddings_for_experts.begin(),
                 selected_embeddings_for_experts.end(),
                 double_selected_embeddings_for_experts.begin(),
                 [](uint64_t val) { return val * 1.0 / std::pow(2, 64); });
  std::transform(experts_weights_one.begin(), experts_weights_one.end(),
                 double_experts_weights_one.begin(),
                 [](uint64_t val) { return val * 1.0 / std::pow(2, 64); });
  std::transform(experts_weights_two.begin(), experts_weights_two.end(),
                 double_experts_weights_two.begin(),
                 [](uint64_t val) { return val * 1.0 / std::pow(2, 64); });

  // Reduce value range to prevent overflow in MPC arithmetic
  // for (auto& val : selected_embeddings_for_experts) {
  //   val = val >> 50;
  // }
  // for (auto& val : experts_weights_one) {
  //   val = val >> 50;
  // }
  // for (auto& val : experts_weights_two) {
  //   val = val >> 50;
  // }

  // Prepare zero inputs for Party 1 (Client/User) - used for secret sharing
  std::vector<double> _selected_embeddings_for_experts(
      params.expert_nums * params.dispatch_expert_per_token *
          params.embedding_size,
      0);

  // Compute expected result using plaintext reference
  auto answer = PlaintextSiLUExpertCompute(
      params, double_selected_embeddings_for_experts,
      double_experts_weights_one, double_experts_weights_two);

  // Launch Party 0 (Server) computation asynchronously
  auto task0 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[0]);
    SiLUExpertComputeServer server(config);
    server.InitEnv(std::move(env));
    server.FeedInputs(double_selected_embeddings_for_experts,
                      double_experts_weights_one, double_experts_weights_two);
    server.Compute();
    return server.ExtractOutput();
  });

  // Launch Party 1 (Client) computation asynchronously
  auto task1 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[1]);
    SiLUExpertComputeUser client(config);
    client.InitEnv(std::move(env));
    client.FeedInputs(_selected_embeddings_for_experts);
    client.Compute();
    return client.ExtractOutput();
  });

  // Wait for both parties to complete and retrieve results
  auto selected_embeddings_for_experts_0 = task0.get();
  auto selected_embeddings_for_experts_1 = task1.get();

  auto isClose = [](double a, double b, double epsilon) -> bool {
    bool flag = std::abs(a - b) < epsilon;
    if (!flag) {
      SPDLOG_INFO("a: {}, b: {}, epsilon: {}", a, b, epsilon);
    }
    return flag;
  };

  // Verify results: both parties should get identical outputs
  for (size_t i = 0; i < params.expert_nums; ++i) {
    for (size_t j = 0; j < params.dispatch_expert_per_token; ++j) {
      size_t offset =
          (i * params.dispatch_expert_per_token + j) * params.embedding_size;

      // Check consistency between Party 0 and Party 1
      // EXPECT_EQ(selected_embeddings_for_experts_0[offset],
      //           selected_embeddings_for_experts_1[offset]);
      EXPECT_TRUE(isClose(selected_embeddings_for_experts_0[offset],
                          selected_embeddings_for_experts_1[offset], 1e-1));

      // // Validate against plaintext reference (skip if slot is empty)
      if (answer[offset] != 0) {
        // EXPECT_EQ(selected_embeddings_for_experts_0[offset], answer[offset]);
        EXPECT_TRUE(isClose(selected_embeddings_for_experts_0[offset],
                            answer[offset], 1e-1));
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, SiLUExpertComputeTest,
    testing::Values(SiLUExpertComputeTestParams{64, 2, 64, 64},
                    SiLUExpertComputeTestParams{64, 2, 64, 128},
                    SiLUExpertComputeTestParams{64, 2, 64, 256}
                    // ,SiLUExpertComputeTestParams{64, 12, 2048, 1408}
                    ));

}  // namespace capricorn

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
