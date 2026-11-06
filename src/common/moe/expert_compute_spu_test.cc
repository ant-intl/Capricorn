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

#include "src/common/moe/expert_compute_spu.h"

#include <cmath>
#include <cstdint>
#include <future>
#include <vector>

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

struct BatchExpertComputeTestParams {
  size_t expert_nums;
  size_t dispatch_expert_per_token{16};
  size_t embedding_size{64};
  size_t expert_dimension{128};
};

class BatchExpertComputeTest
    : public ::testing::TestWithParam<BatchExpertComputeTestParams> {};

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

// SILU activation using piecewise polynomial approximation (matches MPC impl):
//   x < -8.0      : 0
//   x in [-8, -4) : seg1 = a2*x^2 + a1*x + a0
//   x in [-4, 4)  : seg2 = b4*x^6 + b3*x^4 + b2*x^2 + b1*x + b0
//   x >= 4.0      : x
xt::xarray<double> silu(const xt::xarray<double>& x) {
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

std::vector<double> PlaintextBatchExpertCompute(
    const BatchExpertComputeTestParams& param,
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
    auto xt_slice3_activated = silu(xt_slice3);

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

TEST(BatchExpertComputeKernelTest, GeluConsumesBothExpertBiases) {
  auto lctxs = Mock2PCLink(false);
  const BatchExpertComputeConfig config(
      /*expert_nums=*/2, /*token_capacity_per_expert=*/2,
      /*embedding_dimension=*/2, /*expert_dimension=*/2);
  const std::vector<double> selected(8, 0.0);
  const std::vector<double> w1(8, 0.0);
  const std::vector<double> w2{0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25};
  const std::vector<double> b1(4, 0.5);
  const std::vector<double> b2(4, 0.125);

  auto run_party = [&](size_t rank) {
    Environment environment(lctxs[rank]);
    auto selected_copy = selected;
    environment.FeedShare("gelu_selected", xt::adapt(selected_copy, {2, 2, 2}));
    if (rank == 0) {
      auto w1_copy = w1;
      auto b1_copy = b1;
      auto w2_copy = w2;
      auto b2_copy = b2;
      environment.SetPrivate("gelu_w1", xt::adapt(w1_copy, {2, 2, 2}));
      environment.SetPrivate("gelu_b1", xt::adapt(b1_copy, {2, 2}));
      environment.SetPrivate("gelu_w2", xt::adapt(w2_copy, {2, 2, 2}));
      environment.SetPrivate("gelu_b2", xt::adapt(b2_copy, {2, 2}));
    } else {
      environment.GetPrivate("gelu_w1");
      environment.GetPrivate("gelu_b1");
      environment.GetPrivate("gelu_w2");
      environment.GetPrivate("gelu_b2");
    }
    auto output = BatchExpertComputeKernelWithBias(
        environment.spu_ctx_.get(),
        environment.cio_->deviceGetVar("gelu_selected"),
        environment.cio_->deviceGetVar("gelu_w1"),
        environment.cio_->deviceGetVar("gelu_b1"),
        environment.cio_->deviceGetVar("gelu_w2"),
        environment.cio_->deviceGetVar("gelu_b2"), config,
        ExpertActivation::kGelu);
    EXPECT_TRUE(output.isSecret());
    EXPECT_EQ(output.shape(), spu::Shape({2, 2, 2}));
    auto revealed =
        spu::kernel::hal::reveal(environment.spu_ctx_.get(), output);
    auto values = spu::kernel::hal::dump_public_as<double>(
        environment.spu_ctx_.get(), revealed);
    return std::vector<double>(values.begin(), values.end());
  };

  auto party_zero = std::async(std::launch::async, run_party, 0);
  auto result_one = run_party(1);
  auto result_zero = party_zero.get();
  ASSERT_EQ(result_zero.size(), 8);
  ASSERT_EQ(result_zero, result_one);
  const double gelu = 0.5 * 0.5 * (1.0 + std::erf(0.5 / std::sqrt(2.0)));
  const double expected = 0.25 * gelu + 0.125;
  // Allow the three-segment GELU approximation and fixed-point truncation.
  // This tolerance still rejects SiLU substitution (about 0.0086 difference)
  // and omission of either bias (at least 0.08 difference).
  for (double actual : result_zero) {
    EXPECT_NEAR(actual, expected, 0.003);
  }
}

TEST_P(BatchExpertComputeTest, Work) {
  // Setup 2-party communication channels
  auto lctxs = Mock2PCLink(false);
  auto params = GetParam();

  BatchExpertComputeConfig config(
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
  auto answer = PlaintextBatchExpertCompute(
      params, double_selected_embeddings_for_experts,
      double_experts_weights_one, double_experts_weights_two);

  // Launch Party 0 (Server) computation asynchronously
  auto task0 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[0]);
    BatchExpertComputeServer server(config);
    server.InitEnv(std::move(env));
    server.FeedInputs(double_selected_embeddings_for_experts,
                      double_experts_weights_one, double_experts_weights_two);
    server.Compute();
    return server.ExtractOutput();
  });

  // Launch Party 1 (Client) computation asynchronously
  auto task1 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[1]);
    BatchExpertComputeUser client(config);
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
    Works_Instances, BatchExpertComputeTest,
    testing::Values(BatchExpertComputeTestParams{64, 2, 64, 64},
                    BatchExpertComputeTestParams{64, 2, 64, 128},
                    BatchExpertComputeTestParams{64, 2, 64, 256}
                    // ,BatchExpertComputeTestParams{64, 12, 2048, 1408}
                    ));

}  // namespace capricorn

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
