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

#include "src/common/moe/combine.h"

#include <sys/types.h>

#include <cmath>
#include <cstdint>
#include <iterator>

#include "gtest/gtest.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/test_utils.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xcsv.hpp"
#include "xtensor/views/xview.hpp"

#include "yacl/crypto/rand/rand.h"

namespace capricorn {

// Test parameters for the dispatch operation
// This struct defines the configuration for testing the MoE (Mixture of
// Experts) dispatch mechanism, which routes tokens to appropriate experts based
// on scores.
struct CombineTestParams {
  // default 16 (8-64)
  size_t token_nums{16};      // Number of input tokens (batch size)
  size_t embedding_size{64};  // Dimension of each token embedding
  size_t expert_nums;         // Total number of experts in the system
  size_t active_expert_nums;  // Number of experts activated per token (top-k)
  size_t dispatch_expert_per_token;  // Max tokens to dispatch to each expert
  bool deterministic_case{false};

  // deepseek 64 top-k 6, t = 2 m k / n
  // QWen 60 top-k 4, t = 2 m k / n
};

class CombineTest : public ::testing::TestWithParam<CombineTestParams> {};

// Generates random expert indices for testing
// For each token, randomly selects 'active_expert_nums' distinct experts from
// the available 'expert_num' experts. This simulates the routing decision
// made by the gating network in an MoE model.
//
// Args:
//   token_nums: Number of tokens to generate indices for
//   expert_num: Total number of available experts
//   active_expert_nums: Number of experts to select per token
//
// Returns:
//   A flattened vector of expert indices with size token_nums *
//   active_expert_nums
std::vector<uint64_t> MockRandOnthotMatrix(size_t token_nums,
                                           size_t expert_nums,
                                           size_t dispatch_expert_per_token) {
  // Create a list of all expert indices [0, 1, 2, ..., expert_num-1]
  std::vector<uint64_t> origin_indices(token_nums, 0);
  for (size_t i = 0; i < token_nums; i++) {
    origin_indices[i] = i;
  }

  std::random_device rd;
  std::vector<uint64_t> onehot_matrix;

  std::vector<uint64_t> zeros(token_nums * dispatch_expert_per_token, 0);

  for (size_t i = 0; i < expert_nums; i++) {
    std::shuffle(origin_indices.begin(), origin_indices.end(),
                 std::mt19937(rd()));
    auto copy = zeros;
    for (size_t j = 0; j < dispatch_expert_per_token; ++j) {
      copy[j * token_nums + origin_indices[j]] = 1;
    }
    std::copy(copy.begin(), copy.end(), std::back_inserter(onehot_matrix));
  }
  return onehot_matrix;
}

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

std::vector<double> PlaintextCombine(
    const CombineTestParams& param,
    const std::vector<double>& experts_out_embeddings,
    const std::vector<double>& top_t_scores_for_experts,
    const std::vector<uint64_t>& one_hot_token_indices_for_experts) {
  int64_t n = param.expert_nums;
  int64_t m = param.token_nums;
  // int64_t k = param.active_expert_nums;
  int64_t t = param.dispatch_expert_per_token;
  int64_t d = param.embedding_size;

  SPDLOG_INFO("n: {}, m: {}, t: {}, d: {}", n, m, t, d);

  xt::xarray<double> xt_combined_embeddings =
      xt::zeros<double>({uint64_t(m), uint64_t(d)});
  for (int64_t i = 0; i < n; ++i) {
    auto expert_out_embeddings =
        std::vector<double>(experts_out_embeddings.begin() + i * t * d,
                            experts_out_embeddings.begin() + (i + 1) * t * d);
    auto onehot_token_indices = std::vector<double>(
        one_hot_token_indices_for_experts.begin() + i * t * m,
        one_hot_token_indices_for_experts.begin() + (i + 1) * t * m);
    auto top_t_scores =
        std::vector<double>(top_t_scores_for_experts.begin() + i * t,
                            top_t_scores_for_experts.begin() + (i + 1) * t);

    for (int64_t slot = 0; slot < t; ++slot) {
      for (int64_t token = 0; token < m; ++token) {
        if (onehot_token_indices[slot * m + token] == 0.0) {
          continue;
        }
        for (int64_t dim = 0; dim < d; ++dim) {
          xt_combined_embeddings(token, dim) +=
              top_t_scores[slot] * expert_out_embeddings[slot * d + dim];
        }
      }
    }
  }
  SPDLOG_INFO("xt_combined_embeddings: {} x {} : {}",
              xt_combined_embeddings.shape()[0],
              xt_combined_embeddings.shape()[1],
              std::vector<double>(xt_combined_embeddings.begin(),
                                  xt_combined_embeddings.end()));
  return std::vector<double>(xt_combined_embeddings.begin(),
                             xt_combined_embeddings.end());
}

// Main test case: Verifies the MPC-based dispatch implementation against
// plaintext
//
// This test sets up a 2-party computation (2PC) environment where:
// - DispatchServer (Party 0): Provides actual input data
// - DispatchUser (Party 1): Provides zero inputs (placeholder)
//
// The test compares the MPC output from both parties to ensure consistency,
// and validates against the plaintext reference implementation.
TEST_P(CombineTest, Work) {
  // Setup 2-party communication channels
  auto lctxs = Mock2PCLink(false);
  auto params = GetParam();

  if (params.dispatch_expert_per_token == 0) {
    params.dispatch_expert_per_token =
        2 * params.token_nums * params.active_expert_nums / params.expert_nums;
  }

  // Create dispatch configuration
  CombineConfig config(params.token_nums, params.expert_nums,
                       params.dispatch_expert_per_token, params.embedding_size);

  // Prepare test inputs for Party 0 (Server)
  auto experts_out_embeddings = yacl::crypto::RandVec<uint64_t>(
      params.expert_nums * params.dispatch_expert_per_token *
      params.embedding_size);
  auto top_t_scores_for_experts = yacl::crypto::RandVec<uint64_t>(
      params.expert_nums * params.dispatch_expert_per_token);
  auto one_hot_token_indices_for_experts = MockRandOnthotMatrix(
      params.token_nums, params.expert_nums, params.dispatch_expert_per_token);

  auto double_experts_out_embeddings =
      std::vector<double>(experts_out_embeddings.size(), 0);
  auto double_top_t_scores_for_experts =
      std::vector<double>(top_t_scores_for_experts.size(), 0);
  // Reduce value range to prevent overflow in MPC arithmetic
  std::transform(experts_out_embeddings.begin(), experts_out_embeddings.end(),
                 double_experts_out_embeddings.begin(),
                 [](uint64_t val) { return val * 1.0 / std::pow(2, 60); });
  std::transform(top_t_scores_for_experts.begin(),
                 top_t_scores_for_experts.end(),
                 double_top_t_scores_for_experts.begin(),
                 [](uint64_t val) { return val * 1.0 / std::pow(2, 62); });

  if (params.deterministic_case) {
    // Distinct fractional scores, negative outputs, a zero-score slot,
    // repeated destinations within/across experts, and an empty token.
    // Slot destinations: expert 0 -> [2, 0, 2], expert 1 -> [2, 1, 0].
    double_experts_out_embeddings = {
        2.0, -4.0, 6.0, 100.0, 200.0, 300.0, -1.0, 3.0, -5.0,
        4.0, 8.0, -12.0, -2.0, 4.0, 8.0, 6.0, -10.0, 2.0};
    double_top_t_scores_for_experts = {0.5, 0.0, 1.0, 0.25, 0.75, 0.5};
    one_hot_token_indices_for_experts = {
        0, 0, 1, 0,  1, 0, 0, 0,  0, 0, 1, 0,
        0, 0, 1, 0,  0, 1, 0, 0,  1, 0, 0, 0};
  }

  // Reduce value range to prevent overflow in MPC arithmetic
  // for (auto& val : experts_out_embeddings) {
  //   val = val >> 40;
  // }
  // for (auto& val : top_t_scores_for_experts) {
  //   val = val >> 60;
  // }

  // Prepare zero inputs for Party 1 (Client/User) - used for secret sharing
  std::vector<double> _experts_out_embeddings(
      params.expert_nums * params.dispatch_expert_per_token *
          params.embedding_size,
      0);
  std::vector<double> _top_t_scores_for_experts(
      params.expert_nums * params.dispatch_expert_per_token, 0);
  std::vector<uint64_t> _one_hot_token_indices_for_experts(
      params.token_nums * params.expert_nums * params.dispatch_expert_per_token,
      0);

  SPDLOG_INFO("Finding Answer");
  // Compute expected result using plaintext reference
  [[maybe_unused]] auto answer = PlaintextCombine(
      params, double_experts_out_embeddings, double_top_t_scores_for_experts,
      one_hot_token_indices_for_experts);

  // Launch Party 0 (Server) computation asynchronously
  auto task0 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[0]);
    CombineServer server(config);
    server.InitEnv(std::move(env));
    server.FeedInputs(double_experts_out_embeddings,
                      double_top_t_scores_for_experts,
                      one_hot_token_indices_for_experts);
    server.Compute();
    return server.ExtractOutput();
  });

  // Launch Party 1 (Client) computation asynchronously
  auto task1 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[1]);
    CombineUser client(config);
    client.InitEnv(std::move(env));
    client.FeedInputs(_experts_out_embeddings, _top_t_scores_for_experts,
                      _one_hot_token_indices_for_experts);
    client.Compute();
    return client.ExtractOutput();
  });

  // Wait for both parties to complete and retrieve results
  auto combined_embeddings_for_experts_0 = task0.get();
  auto combined_embeddings_for_experts_1 = task1.get();

  ASSERT_EQ(combined_embeddings_for_experts_0.size(), answer.size());
  ASSERT_EQ(combined_embeddings_for_experts_1.size(), answer.size());
  // Check every output dimension, including zeros for unselected tokens.
  for (size_t i = 0; i < answer.size(); ++i) {
    EXPECT_NEAR(combined_embeddings_for_experts_0[i],
                combined_embeddings_for_experts_1[i], 1e-2) << "index " << i;
    EXPECT_NEAR(combined_embeddings_for_experts_0[i], answer[i], 1e-2)
        << "index " << i;
  }
}

// Instantiate test cases with different parameter combinations
// Each tuple represents: {token_nums, embedding_size, expert_nums,
// active_expert_nums, dispatch_expert_per_token}
//
// Test matrix covers:
// - Token counts: 16, 32, 64
// - Embedding sizes: 256, 1024
// - Fixed: 64 experts, top-10 expert selection, 2 tokens per expert dispatch
// limit
INSTANTIATE_TEST_SUITE_P(
    Works_Instances, CombineTest,
    testing::Values(
        CombineTestParams{4, 3, 2, 2, 3, true},
        // ------- DEEPSEEK
        CombineTestParams{8, 64, 64, 6, 0},   // Small batch, small embedding
        CombineTestParams{16, 64, 64, 6, 0},  // Small batch, small embedding
        CombineTestParams{32, 64, 64, 6, 0},  // Small batch, small embedding
        CombineTestParams{64, 64, 64, 6, 0},  // Small batch, small embedding
        // -------- QWEN
        CombineTestParams{8, 64, 60, 4, 0},   // Small batch, small embedding
        CombineTestParams{16, 64, 60, 4, 0},  // Small batch, small embedding
        CombineTestParams{32, 64, 60, 4, 0},  // Small batch, small embedding
        CombineTestParams{64, 64, 60, 4, 0}   // Small batch, small embedding
        ));

}  // namespace capricorn

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
