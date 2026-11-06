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

#include "src/capricorn/routing.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"
#include "spdlog/spdlog.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/test_utils.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"

#include "yacl/crypto/rand/rand.h"

namespace capricorn {

struct RoutingTestParams {
  size_t m;  // token nums
  size_t d;  // hidden dimension
  size_t N;  // expert nums
  size_t t;  // expert capacity (top-t tokens per expert)
};

class RoutingTest : public ::testing::TestWithParam<RoutingTestParams> {
 protected:
  void SetUp() override { SPDLOG_INFO("Setting up RoutingTest"); }
};

std::pair<std::vector<double>, std::vector<double>> PlaintextRouting(
    const RoutingConfig& config, const xt::xarray<double>& token_embeddings,
    const xt::xarray<double>& gate_weights) {
  // Adapt vectors to xtensor arrays
  // x: [m, d], W: [n, d]
  // auto x = xt::adapt(token_embeddings, {config.m_, config.d_});
  // auto W = xt::adapt(gate_weights, {config.m_, config.d_});
  auto& x = token_embeddings;
  auto& W = gate_weights;

  // Step 1: Linear transformation - compute logits = x @ W^T
  // x: [m, d], W^T: [d, n] -> logits: [m, n]
  auto W_T = xt::transpose(W, {1, 0});

  // Manual matrix multiplication: logits = x @ W_T
  // logits[i][j] = sum_k(x[i][k] * W_T[k][j])
  xt::xarray<double> logits = xt::zeros<double>({config.m_, config.n_});
  for (size_t i = 0; i < config.m_; ++i) {
    for (size_t j = 0; j < config.n_; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < config.d_; ++k) {
        sum += x(i, k) * W_T(k, j);
      }
      logits(i, j) = sum;
    }
  }

  // Step 2: Softmax over expert dimension (axis=1)
  // Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
  // Compute max over axis=1 for numerical stability
  auto max_vals = xt::amax(logits, {1});  // Shape: [m]

  // Subtract max (broadcasted) and compute exp
  auto shifted = logits - xt::view(max_vals, xt::all(), xt::newaxis());
  auto exp_shifted = xt::exp(shifted);

  // Compute sum of exp over axis=1
  auto sum_exp = xt::sum(exp_shifted, {1});  // Shape: [m]

  // Divide to get softmax scores
  // Handle edge case where sum_exp is 0 (all exp values underflowed)
  xt::xarray<double> scores = xt::zeros<double>({config.m_, config.n_});
  for (size_t i = 0; i < config.m_; ++i) {
    if (sum_exp(i) > 0.0) {
      // Normal case: divide by sum
      for (size_t j = 0; j < config.n_; ++j) {
        scores(i, j) = exp_shifted(i, j) / sum_exp(i);
      }
    } else {
      // Edge case: all exp values underflowed, use uniform distribution
      for (size_t j = 0; j < config.n_; ++j) {
        scores(i, j) = 1.0 / config.n_;
      }
    }
  }

  // Step 3: For each expert, select its top-t tokens (Dense MoE routing).
  std::vector<double> selected_embeddings(config.n_ * config.t_ * config.d_,
                                          0.0);
  std::vector<double> selected_token_scores(config.n_ * config.t_, 0.0);
  for (size_t j = 0; j < config.n_; ++j) {
    std::map<double, size_t, std::greater<double>> expert_scores;
    for (size_t i = 0; i < config.m_; ++i) {
      expert_scores[scores(i, j)] = i;
    }

    size_t counter = 0;
    for (auto& it : expert_scores) {
      if (counter >= config.t_) {
        break;
      }
      auto idx = it.second;
      // SPDLOG_INFO("Expert {}: Token {} -> Score {}", j, idx, it.first);
      for (size_t k = 0; k < config.d_; ++k) {
        selected_embeddings[(j * config.t_ + counter) * config.d_ + k] =
            x(idx, k);
      }
      selected_token_scores[j * config.t_ + counter] = it.first;
      counter++;
    }
  }

  return std::make_pair(selected_embeddings, selected_token_scores);
}

TEST(ExpertChoiceRoutingKernelTest, SelectsTopTokensPerExpert) {
  auto lctxs = Mock2PCLink(false);
  const RoutingConfig config(/*token_nums=*/4, /*expert_nums=*/2,
                             /*expert_capacity=*/2,
                             /*embedding_dimension=*/2);
  const std::vector<double> token_embeddings = {-2.0, 0.0, -1.0, 0.0,
                                                1.0,  0.0, 2.0,  0.0};
  const std::vector<double> zero_token_embeddings(8, 0.0);
  const std::vector<double> gate_weights = {1.0, 0.0, -1.0, 0.0};

  auto run_party = [&](size_t rank) {
    Environment environment(lctxs[rank]);
    auto token_data = rank == 0 ? token_embeddings : zero_token_embeddings;
    auto token_tensor = xt::adapt(token_data, {4, 2});
    environment.FeedShare(config.kTokenEmbeddings, token_tensor);

    if (rank == 0) {
      auto gate_data = gate_weights;
      auto gate_tensor = xt::adapt(gate_data, {2, 2});
      environment.SetPrivate(config.kGateWeights, gate_tensor);
    } else {
      environment.GetPrivate(config.kGateWeights);
    }

    auto result = ExpertChoiceRoutingKernel(
        environment.spu_ctx_.get(),
        environment.cio_->deviceGetVar(config.kTokenEmbeddings),
        environment.cio_->deviceGetVar(config.kGateWeights), config);
    EXPECT_EQ(result.selected_embeddings.shape(), spu::Shape({2, 2, 2}));
    EXPECT_EQ(result.selected_scores.shape(), spu::Shape({2, 2}));
    EXPECT_EQ(result.token_one_hot.shape(), spu::Shape({2, 2, 4}));
    EXPECT_TRUE(result.selected_embeddings.isSecret());
    EXPECT_TRUE(result.selected_scores.isSecret());
    EXPECT_TRUE(result.token_one_hot.isSecret());

    auto revealed = spu::kernel::hal::reveal(environment.spu_ctx_.get(),
                                             result.token_one_hot);
    auto values = spu::kernel::hal::dump_public_as<uint64_t>(
        environment.spu_ctx_.get(), revealed);
    return std::vector<uint64_t>(values.begin(), values.end());
  };

  auto party_zero = std::async(std::launch::async, run_party, 0);
  auto party_one_result = run_party(1);
  auto party_zero_result = party_zero.get();

  const std::vector<uint64_t> expected = {
      0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
  };
  EXPECT_EQ(party_zero_result, expected);
  EXPECT_EQ(party_one_result, expected);
}

TEST_P(RoutingTest, BasicRouting) {
  auto params = GetParam();
  SPDLOG_INFO("Testing Routing with m={}, d={}, N={}, t={}", params.m, params.d,
              params.N, params.t);

  // Create mock link
  auto lctxs = Mock2PCLink(false);

  ASSERT_GE(params.m, 2);
  ASSERT_GE(params.d, 1);
  ASSERT_EQ(params.N % 2, 0);

  // Dense MoE Expert-Choice fixture. Hidden states are signed and bounded,
  // while paired gate vectors keep logits in [-1, 1]. The first half of the
  // experts deterministically select token 0 and the second half select the
  // last token, with a non-zero Top-1 margin even for m=128.
  std::vector<double> double_token_embeddings(params.m * params.d, 0.0);
  for (size_t token = 0; token < params.m; ++token) {
    const double routing_coordinate =
        -1.0 +
        2.0 * static_cast<double>(token) / static_cast<double>(params.m - 1);
    double_token_embeddings[token * params.d] = routing_coordinate;
    for (size_t dim = 1; dim < params.d; ++dim) {
      double_token_embeddings[token * params.d + dim] =
          (token + dim) % 2 == 0 ? 1.0 : -1.0;
    }
  }

  std::vector<double> double_gate_weights(params.N * params.d, 0.0);
  for (size_t expert = 0; expert < params.N; ++expert) {
    double_gate_weights[expert * params.d] = expert < params.N / 2 ? -1.0 : 1.0;
  }

  // Zero shares for user
  std::vector<double> _token_embeddings(params.m * params.d, 0.0f);

  // Expected output sizes
  size_t expected_embeddings_size = params.N * params.t * params.d;

  RoutingConfig config(params.m, params.N, params.t, params.d);

  auto expected = PlaintextRouting(
      config, xt::adapt(double_token_embeddings, {params.m, params.d}),
      xt::adapt(double_gate_weights, {params.N, params.d}));
  const auto& expect_embedding = expected.first;

  for (size_t expert = 0; expert < params.N; ++expert) {
    const double expected_coordinate = expert < params.N / 2 ? -1.0 : 1.0;
    EXPECT_DOUBLE_EQ(expect_embedding[expert * params.t * params.d],
                     expected_coordinate);
  }

  // Run server and user concurrently
  auto server_task = std::async([&]() {
    RoutingServer server(config);
    auto env = std::make_unique<Environment>(lctxs[0]);
    server.InitEnv(std::move(env));

    server.FeedInputs(double_token_embeddings, double_gate_weights);
    server.Compute();
    return server.ExtractOutput();
  });

  auto user_task = std::async([&]() {
    RoutingUser user(config);
    auto env = std::make_unique<Environment>(lctxs[1]);
    user.InitEnv(std::move(env));

    user.FeedInputs(_token_embeddings);
    user.Compute();
    return user.ExtractOutput();
  });

  // Get results
  auto server_result = server_task.get();
  auto user_result = user_task.get();

  // Verify output sizes
  EXPECT_EQ(server_result.size(), expected_embeddings_size);
  EXPECT_EQ(user_result.size(), expected_embeddings_size);

  SPDLOG_INFO("Server embeddings size: {}", server_result.size());
  SPDLOG_INFO("User embeddings size: {}", user_result.size());

  // Verify server and user have the same results
  for (size_t i = 0; i < server_result.size(); ++i) {
    EXPECT_DOUBLE_EQ(server_result[i], user_result[i])
        << "Mismatch at index " << i;
  }

  for (size_t i = 0; i < server_result.size(); ++i) {
    EXPECT_NEAR(server_result[i], expect_embedding[i], 1e-3)
        << "Mismatch at flattened output index " << i;
  }
}

// Test with small dimensions
INSTANTIATE_TEST_SUITE_P(SmallDimensions, RoutingTest,
                         ::testing::Values(RoutingTestParams{4, 16, 64, 1},
                                           RoutingTestParams{8, 16, 64, 1},
                                           RoutingTestParams{16, 16, 64, 1},
                                           RoutingTestParams{32, 16, 64, 1},
                                           RoutingTestParams{64, 16, 64, 1},
                                           RoutingTestParams{128, 16, 64, 1}));

}  // namespace capricorn

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
