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

#include "src/common/moe/token_topk.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/test_utils.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"

#include "yacl/crypto/rand/rand.h"

namespace capricorn {

struct GateRoutingTestParams {
  size_t token_nums;
  size_t expert_nums;
  size_t top_k;
  size_t embedding_dim;
};

class GateRoutingTest : public ::testing::TestWithParam<GateRoutingTestParams> {
};

// Plaintext reference implementation of gate routing using xtensor
// Computes: W, K = Top-K(Softmax(Linear(x)), k)

TEST_P(GateRoutingTest, Work) {
  auto lctxs = Mock2PCLink(false);

  auto params = GetParam();

  GateRoutingConfig config(params.token_nums, params.expert_nums, params.top_k,
                           params.embedding_dim);

  // Prepare inputs
  auto token_embeddings =
      yacl::crypto::RandVec<uint64_t>(params.token_nums * params.embedding_dim);
  auto gate_weights = yacl::crypto::RandVec<uint64_t>(params.expert_nums *
                                                      params.embedding_dim);

  auto double_token_embeddings =
      std::vector<double>(token_embeddings.size(), 0.0);
  auto double_gate_weights = std::vector<double>(gate_weights.size(), 0.0);

  // Scale to avoid overflow in MPC while preventing underflow in exp
  // Use smaller scaling factor to keep values in a reasonable range
  std::transform(token_embeddings.begin(), token_embeddings.end(),
                 double_token_embeddings.begin(),
                 [](uint64_t val) { return val * 1.0 / std::pow(2, 63); });
  std::transform(gate_weights.begin(), gate_weights.end(),
                 double_gate_weights.begin(),
                 [](uint64_t val) { return val * 1.0 / std::pow(2, 63); });
  //   for (auto& val : token_embeddings) {
  //     val = val / 100.0;
  //   }
  //   for (auto& val : gate_weights) {
  //     val = val / 100.0;
  //   }

  // Zero shares for user
  std::vector<double> _token_embeddings(
      params.token_nums * params.embedding_dim, 0.0);

  // Compute expected result
  // auto [expected_scores, expected_indices] = PlaintextGateRouting(
  //     params, double_token_embeddings, double_gate_weights);

  // Run MPC computation
  auto task0 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[0]);
    GateRoutingServer server(config);
    server.InitEnv(std::move(env));
    server.FeedInputs(double_token_embeddings, double_gate_weights);
    server.Compute();
    server.ExtractOutput();
    server._ExtractExpectedOutput();
    return;
  });

  // auto task1 = std::async([&] {
  auto env = std::make_unique<Environment>(lctxs[1]);
  GateRoutingUser client(config);
  client.InitEnv(std::move(env));
  client.FeedInputs(_token_embeddings);
  client.Compute();
  auto [mpc_scores, mpc_indices] = client.ExtractOutput();
  auto [_scores, _indices] = client._ExtractExpectedOutput();

  task0.get();

  for (size_t i = 0; i < params.token_nums; ++i) {
    SPDLOG_INFO(
        "Expected Expert Indices: {}",
        std::vector<uint64_t>(_indices.begin() + i * params.top_k,
                              _indices.begin() + (i + 1) * params.top_k));
    SPDLOG_INFO(
        "Expert Indices: {}",
        std::vector<uint64_t>(mpc_indices.begin() + i * params.top_k,
                              mpc_indices.begin() + (i + 1) * params.top_k));
    SPDLOG_INFO("Expected Expert Scores: {}",
                std::vector<double>(_scores.begin() + i * params.top_k,
                                    _scores.begin() + (i + 1) * params.top_k));
    SPDLOG_INFO(
        "Expert Scores: {}",
        std::vector<double>(mpc_scores.begin() + i * params.top_k,
                            mpc_scores.begin() + (i + 1) * params.top_k));
  }

  EXPECT_TRUE(mpc_scores.size() == _scores.size());
  EXPECT_TRUE(mpc_indices.size() == _indices.size());
  EXPECT_TRUE(mpc_scores.size() == mpc_indices.size());

  for (size_t i = 0; i < mpc_scores.size(); ++i) {
    // Dispatch's M + W ranking key requires the Softmax output contract.
    EXPECT_GE(mpc_scores[i], 0.0);
    EXPECT_LE(mpc_scores[i], 1.0);
    EXPECT_NEAR(mpc_scores[i], _scores[i], 0.01);
    if (_scores[i] > 0.01) {
      EXPECT_EQ(mpc_indices[i], _indices[i]);
    }
  }

  // return client.ExtractOutput();
  // });

  // auto [scores_0, indices_0] = task0.get();
  // auto [scores_1, indices_1] = task1.get();

  // for (size_t i = 0; i < params.token_nums; ++i) {
  //   SPDLOG_INFO("Expected Expert Indices: {}",
  //               std::vector<uint64_t>(
  //                   expected_indices.begin() + i * params.top_k,
  //                   expected_indices.begin() + (i + 1) * params.top_k));
  //   SPDLOG_INFO(
  //       "Expert Indices: {}",
  //       std::vector<uint64_t>(indices_0.begin() + i * params.top_k,
  //                             indices_0.begin() + (i + 1) * params.top_k));
  //   SPDLOG_INFO(
  //       "Expected Expert Scores: {}",
  //       std::vector<double>(expected_scores.begin() + i * params.top_k,
  //                           expected_scores.begin() + (i + 1) *
  //                           params.top_k));
  //   SPDLOG_INFO("Expert Scores: {}",
  //               std::vector<double>(scores_0.begin() + i * params.top_k,
  //                                   scores_0.begin() + (i + 1) *
  //                                   params.top_k));
  // }

  // // Verify both parties get the same result
  // ASSERT_EQ(scores_0.size(), scores_1.size());
  // ASSERT_EQ(indices_0.size(), indices_1.size());

  // for (size_t i = 0; i < scores_0.size(); ++i) {
  //   EXPECT_DOUBLE_EQ(scores_0[i], scores_1[i]);
  // }

  // // Verify both parties get the same result
  // for (size_t i = 0; i < indices_0.size(); ++i) {
  //   EXPECT_EQ(indices_0[i], indices_1[i]);
  // }

  // // Verify against plaintext (with tolerance for floating point)
  // for (size_t i = 0; i < scores_0.size(); ++i) {
  //   EXPECT_NEAR(scores_0[i], expected_scores[i], 0.01);
  // }

  // // Verify indices match plaintext
  // for (size_t i = 0; i < indices_0.size(); ++i) {
  //   if (expected_scores[i] > 0.01) {
  //     EXPECT_EQ(indices_0[i], expected_indices[i]);
  //   }
  // }
}

INSTANTIATE_TEST_SUITE_P(Works_Instances, GateRoutingTest,
                         testing::Values(GateRoutingTestParams{16, 8, 2, 64},
                                         GateRoutingTestParams{32, 16, 2, 128},
                                         GateRoutingTestParams{64, 32, 2,
                                                               256}));

}  // namespace capricorn

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
