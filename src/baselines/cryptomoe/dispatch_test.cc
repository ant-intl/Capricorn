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

#include "src/baselines/cryptomoe/dispatch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include "gtest/gtest.h"
#include "libspu/kernel/hal/type_cast.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/test_utils.h"

#include "yacl/crypto/rand/rand.h"

namespace capricorn {

constexpr double kScoreTolerance = std::ldexp(8.0, -18);

struct PlainDispatchResult {
  std::vector<double> selected_embeddings;
  std::vector<double> selected_scores;
};

struct SecureDispatchResult {
  std::vector<double> selected_embeddings;
  std::vector<double> selected_scores;
  std::vector<uint64_t> token_one_hot;
};

// Test parameters for the dispatch operation
// This struct defines the configuration for testing the MoE (Mixture of
// Experts) dispatch mechanism, which routes tokens to appropriate experts based
// on scores.
struct DispatchTestParams {
  // default 16 (8-64)
  size_t token_nums{16};      // Number of input tokens (batch size)
  size_t embedding_size{64};  // Dimension of each token embedding
  size_t expert_nums;         // Total number of experts in the system
  size_t active_expert_nums;  // Number of experts activated per token (top-k)
  size_t dispatch_expert_per_token;  // Max tokens to dispatch to each expert
  ExpertTopTMode top_t_mode{ExpertTopTMode::SerialQuickSelect};

  // deepseek 64 top-k 6, t = 2 m k / n
  // QWen 60 top-k 4, t = 2 m k / n
};

class DispatchTest : public ::testing::TestWithParam<DispatchTestParams> {};

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
std::vector<uint64_t> MockRandomIndices(size_t token_nums, size_t expert_num,
                                        size_t active_expert_nums) {
  YACL_ENFORCE(active_expert_nums <= expert_num);

  // Create a list of all expert indices [0, 1, 2, ..., expert_num-1]
  std::vector<uint64_t> origin_indices(expert_num, 0);
  for (size_t i = 0; i < expert_num; i++) {
    origin_indices[i] = i;
  }
  std::random_device rd;
  std::vector<uint64_t> random_indices(token_nums * active_expert_nums, 0);

  // For each token, shuffle experts and pick the first 'active_expert_nums'
  for (size_t i = 0; i < token_nums; i++) {
    std::shuffle(origin_indices.begin(), origin_indices.end(),
                 std::mt19937(rd()));
    auto span = absl::MakeSpan(random_indices)
                    .subspan(i * active_expert_nums, active_expert_nums);
    std::copy(origin_indices.begin(),
              origin_indices.begin() + active_expert_nums, span.begin());
  }
  return random_indices;
}

// Plaintext reference implementation of the dispatch operation
// This function simulates the dispatch mechanism in plaintext for verification
// purposes. It routes token embeddings to experts based on expert indices and
// scores.
//
// The dispatch process works as follows:
// 1. Group tokens by their assigned experts, keeping track of scores
// 2. For each expert, select top-scoring tokens (up to
// dispatch_expert_per_token)
// 3. Copy selected token embeddings to the output buffer organized by expert
//
// Args:
//   param: Configuration parameters for the dispatch operation
//   token_embeddings: Flattened token embeddings [token_nums * embedding_size]
//   expert_indices: Expert assignments for each token [token_nums *
//   active_expert_nums] expert_scores: Scores for each expert assignment
//   [token_nums * active_expert_nums]
//
// Returns:
//   Selected embeddings organized as
//   [expert_nums][dispatch_expert_per_token][embedding_size]
PlainDispatchResult PlaintestDispatch(const DispatchTestParams& param,
                                      std::vector<double>& token_embeddings,
                                      std::vector<uint64_t>& expert_indices,
                                      std::vector<double>& expert_scores) {
  // Keep every candidate, including equal scores. A map keyed by score would
  // silently discard ties and make the oracle unsuitable for boundary tests.
  std::vector<std::vector<std::pair<double, uint64_t>>> expert_scores_token(
      param.expert_nums);

  // Build the expert-to-tokens mapping
  for (size_t i = 0; i < param.token_nums; ++i) {
    for (size_t j = 0; j < param.active_expert_nums; ++j) {
      auto expert_index = expert_indices[i * param.active_expert_nums + j];
      auto expert_score = expert_scores[i * param.active_expert_nums + j];
      expert_scores_token[expert_index].emplace_back(expert_score, i);
    }
  }

  // Output buffer: for each expert, store up to dispatch_expert_per_token
  // embeddings
  std::vector<double> selected_embeddings(param.dispatch_expert_per_token *
                                              param.expert_nums *
                                              param.embedding_size,
                                          0);
  std::vector<double> selected_scores(
      param.dispatch_expert_per_token * param.expert_nums, 0);

  // Process each expert and copy selected token embeddings
  for (size_t i = 0; i < param.expert_nums; ++i) {
    auto& scores_token = expert_scores_token[i];
    std::stable_sort(
        scores_token.begin(), scores_token.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

    size_t counter = 0;
    auto it = scores_token.begin();
    // Select top-scoring tokens for this expert, up to
    // dispatch_expert_per_token limit
    while (counter < param.dispatch_expert_per_token &&
           it != scores_token.end()) {
      auto idx = it->second;

      // Copy the selected token's embedding to the output buffer
      std::copy(token_embeddings.begin() + idx * param.embedding_size,
                token_embeddings.begin() + (idx + 1) * param.embedding_size,
                selected_embeddings.begin() +
                    (i * param.dispatch_expert_per_token + counter) *
                        param.embedding_size);
      selected_scores[i * param.dispatch_expert_per_token + counter] =
          it->first;
      it = std::next(it);
      ++counter;
    }
  }

  return {std::move(selected_embeddings), std::move(selected_scores)};
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
TEST_P(DispatchTest, Work) {
  // Setup 2-party communication channels
  auto lctxs = Mock2PCLink(false);
  auto params = GetParam();

  if (params.dispatch_expert_per_token == 0) {
    params.dispatch_expert_per_token =
        2 * params.token_nums * params.active_expert_nums / params.expert_nums;
  }

  // Create dispatch configuration
  DispatchConfig config(params.token_nums, params.expert_nums,
                        params.active_expert_nums,
                        params.dispatch_expert_per_token, params.embedding_size,
                        params.top_t_mode);

  // Prepare test inputs for Party 0 (Server)
  auto token_embeddings = yacl::crypto::RandVec<uint64_t>(
      params.token_nums * params.embedding_size);

  auto flatten_expert_indices = MockRandomIndices(
      params.token_nums, params.expert_nums, params.active_expert_nums);
  auto flatten_expert_scores = yacl::crypto::RandVec<uint64_t>(
      params.token_nums * params.active_expert_nums);

  auto double_token_embeddings =
      std::vector<double>(token_embeddings.size(), 0);
  auto double_flatten_expert_scores =
      std::vector<double>(flatten_expert_scores.size(), 0);
  // Reduce value range to prevent overflow in MPC arithmetic
  std::transform(token_embeddings.begin(), token_embeddings.end(),
                 double_token_embeddings.begin(),
                 [](uint64_t val) { return val * 1.0 / std::pow(2, 60); });
  std::transform(flatten_expert_scores.begin(), flatten_expert_scores.end(),
                 double_flatten_expert_scores.begin(), [](uint64_t val) {
                   // GateRouting feeds Dispatch with Softmax values in [0, 1].
                   // Keep 52 random bits so conversion to double cannot round
                   // the largest sample up to exactly 1.0.
                   return std::ldexp(static_cast<double>(val >> 12), -52);
                 });
  if (double_flatten_expert_scores.size() >= 5) {
    double_flatten_expert_scores[0] = 0.0;
    double_flatten_expert_scores[1] = std::nextafter(1.0, 0.0);
    double_flatten_expert_scores[2] = 1.0;
    double_flatten_expert_scores[3] = 0.5;
    double_flatten_expert_scores[4] = 0.5;
  }
  // for (auto& val : token_embeddings) {
  //   val = val >> 8;
  // }
  // for (auto& val : flatten_expert_scores) {
  //   val = val >> 8;
  // }

  // Prepare zero inputs for Party 1 (Client/User) - used for secret sharing
  std::vector<double> _token_embeddings(
      params.token_nums * params.embedding_size, 0);
  std::vector<uint64_t> _flatten_expert_indices(
      params.token_nums * params.active_expert_nums, 0);
  std::vector<double> _flatten_expert_scores(
      params.token_nums * params.active_expert_nums, 0);

  // Compute expected result using plaintext reference
  auto answer =
      PlaintestDispatch(params, double_token_embeddings, flatten_expert_indices,
                        double_flatten_expert_scores);

  // Launch Party 0 (Server) computation asynchronously
  auto task0 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[0]);
    DispatchServer server(config);
    server.InitEnv(std::move(env));
    server.FeedInputs(double_token_embeddings, flatten_expert_indices,
                      double_flatten_expert_scores);
    server.Compute();
    auto xt_selected_scores = spu::kernel::hal::dump_public_as<double>(
        server.env_ptr_->spu_ctx_.get(),
        spu::kernel::hal::reveal(
            server.env_ptr_->spu_ctx_.get(),
            server.env_ptr_->cio_->deviceGetVar(config.kTopTScoresForExperts)));
    auto xt_token_one_hot = spu::kernel::hal::dump_public_as<uint64_t>(
        server.env_ptr_->spu_ctx_.get(),
        spu::kernel::hal::reveal(server.env_ptr_->spu_ctx_.get(),
                                 server.env_ptr_->cio_->deviceGetVar(
                                     config.kOneHotTokenIndicesForExperts)));
    return SecureDispatchResult{server.ExtractOutput(),
                                std::vector<double>(xt_selected_scores.begin(),
                                                    xt_selected_scores.end()),
                                std::vector<uint64_t>(xt_token_one_hot.begin(),
                                                      xt_token_one_hot.end())};
  });

  // Launch Party 1 (Client) computation asynchronously
  auto task1 = std::async([&] {
    auto env = std::make_unique<Environment>(lctxs[1]);
    DispatchUser client(config);
    client.InitEnv(std::move(env));
    client.FeedInputs(_token_embeddings, _flatten_expert_indices,
                      _flatten_expert_scores);
    client.Compute();
    auto xt_selected_scores = spu::kernel::hal::dump_public_as<double>(
        client.env_ptr_->spu_ctx_.get(),
        spu::kernel::hal::reveal(
            client.env_ptr_->spu_ctx_.get(),
            client.env_ptr_->cio_->deviceGetVar(config.kTopTScoresForExperts)));
    auto xt_token_one_hot = spu::kernel::hal::dump_public_as<uint64_t>(
        client.env_ptr_->spu_ctx_.get(),
        spu::kernel::hal::reveal(client.env_ptr_->spu_ctx_.get(),
                                 client.env_ptr_->cio_->deviceGetVar(
                                     config.kOneHotTokenIndicesForExperts)));
    return SecureDispatchResult{client.ExtractOutput(),
                                std::vector<double>(xt_selected_scores.begin(),
                                                    xt_selected_scores.end()),
                                std::vector<uint64_t>(xt_token_one_hot.begin(),
                                                      xt_token_one_hot.end())};
  });

  // Wait for both parties to complete and retrieve results
  auto output0 = task0.get();
  auto output1 = task1.get();

  // Verify results: both parties should get identical outputs
  for (size_t i = 0; i < params.expert_nums; ++i) {
    std::vector<double> actual_scores;
    actual_scores.reserve(params.dispatch_expert_per_token);

    for (size_t j = 0; j < params.dispatch_expert_per_token; ++j) {
      const size_t slot = i * params.dispatch_expert_per_token + j;
      const size_t one_hot_offset = slot * params.token_nums;
      size_t selected_token = params.token_nums;
      size_t one_count = 0;
      for (size_t token = 0; token < params.token_nums; ++token) {
        const auto bit0 = output0.token_one_hot[one_hot_offset + token];
        const auto bit1 = output1.token_one_hot[one_hot_offset + token];
        EXPECT_EQ(bit0, bit1) << "expert=" << i << ", slot=" << j;
        EXPECT_TRUE(bit0 == 0 || bit0 == 1);
        if (bit0 == 1) {
          selected_token = token;
          ++one_count;
        }
      }
      ASSERT_EQ(one_count, 1U) << "expert=" << i << ", slot=" << j;

      const size_t embedding_offset = slot * params.embedding_size;
      const size_t token_embedding_offset =
          selected_token * params.embedding_size;
      for (size_t d = 0; d < params.embedding_size; ++d) {
        EXPECT_NEAR(output0.selected_embeddings[embedding_offset + d],
                    output1.selected_embeddings[embedding_offset + d], 1e-3);
        EXPECT_NEAR(output0.selected_embeddings[embedding_offset + d],
                    double_token_embeddings[token_embedding_offset + d], 1e-3)
            << "expert=" << i << ", slot=" << j << ", dim=" << d;
      }

      double expected_slot_score = 0.0;
      for (size_t active = 0; active < params.active_expert_nums; ++active) {
        const size_t candidate =
            selected_token * params.active_expert_nums + active;
        if (flatten_expert_indices[candidate] == i) {
          expected_slot_score = double_flatten_expert_scores[candidate];
          break;
        }
      }
      EXPECT_NEAR(output0.selected_scores[slot], output1.selected_scores[slot],
                  kScoreTolerance);
      // A zero-score padding candidate can come from a different active slot
      // of a token that also happens to route to this expert. Dividing the
      // flattened candidate index by k intentionally drops that slot identity,
      // so only positive-score slots have a unique token/score binding to
      // verify here. The complete score multiset check below still verifies the
      // number of zero padding slots.
      if (output0.selected_scores[slot] != 0.0) {
        EXPECT_NEAR(output0.selected_scores[slot], expected_slot_score,
                    kScoreTolerance)
            << "expert=" << i << ", slot=" << j
            << ", selected_token=" << selected_token;
      }
      actual_scores.push_back(output0.selected_scores[slot]);
    }

    std::vector<double> expected_scores(
        answer.selected_scores.begin() + i * params.dispatch_expert_per_token,
        answer.selected_scores.begin() +
            (i + 1) * params.dispatch_expert_per_token);
    std::sort(actual_scores.begin(), actual_scores.end());
    std::sort(expected_scores.begin(), expected_scores.end());
    for (size_t slot = 0; slot < actual_scores.size(); ++slot) {
      EXPECT_NEAR(actual_scores[slot], expected_scores[slot], kScoreTolerance)
          << "expert=" << i << ", score slot=" << slot;
    }
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
    Works_Instances, DispatchTest,
    testing::Values(
        // ------- DEEPSEEK -------
        DispatchTestParams{8, 64, 64, 6, 0}  // Small batch, small embedding
        ,
        DispatchTestParams{16, 64, 64, 6, 0}  // Small batch, small embedding
        ,
        DispatchTestParams{32, 64, 64, 6, 0}  // Small batch, small embedding
        ,
        DispatchTestParams{64, 64, 64, 6, 0}  // Small batch, small embedding
        // -------- QWEN -------
        ,
        DispatchTestParams{8, 64, 60, 4, 0}  // Small batch, small embedding
        ,
        DispatchTestParams{16, 64, 60, 4, 0}  // Small batch, small embedding
        ,
        DispatchTestParams{32, 64, 60, 4, 0}  // Small batch, small embedding
        ,
        DispatchTestParams{64, 64, 60, 4, 0}  // Small batch, small embedding
        ,
        DispatchTestParams{8, 32, 8, 2, 4,
                           ExpertTopTMode::ParallelQuickSelect}));

}  // namespace capricorn

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
