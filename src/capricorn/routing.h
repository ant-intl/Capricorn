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

#pragma once

#include <memory>
#include <unordered_map>

#include "libspu/core/context.h"
#include "libspu/core/pt_buffer_view.h"
#include "libspu/core/value.h"
#include "libspu/device/io.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/prot_helper.h"
#include "src/common/runtime/prot_interface.h"

namespace capricorn {

// Routing Configuration
// Computes: For each expert E_i, select top-t tokens based on routing scores
// - X: token embeddings [m, d]
// - W: routing scores [m, N] where W = Softmax(X @ GateWeights)
// - For each expert i: select top-t tokens -> X_i [t, d], scores_i [t]
struct RoutingConfig {
  // configure parameters
  size_t m_{0};  // token nums
  size_t n_{0};  // expert nums
  size_t t_{0};  // expert capacity (top-t tokens per expert)
  size_t d_{0};  // embedding dimension

  // variable names in cio
  std::string kTokenEmbeddings{"token_embeddings"};  // [m, d]
  std::string kGateWeights{"gate_weights"};          // [n, d]

  // output variable names in cio
  // output variable name in cio
  std::string kSelectedEmbeddingsForExperts{
      "selected_embeddings_for_experts"};  // should be in size n_ * t_ * d_
  std::string kTopTScoresForExperts{
      "top_t_scores_for_experts"};  // should be in size n_ * t_
  std::string kOneHotTokenIndicesForExperts{
      "one_hot_token_indices_for_experts"};  // should be in size n_ * t_ * m_

  RoutingConfig(size_t token_nums, size_t expert_nums,
                size_t activated_experts_per_token, size_t embedding_dimension)
      : m_(token_nums),
        n_(expert_nums),
        t_(activated_experts_per_token),
        d_(embedding_dimension) {}
};

struct ExpertChoiceRoutingResult {
  spu::Value selected_embeddings;
  spu::Value selected_scores;
  spu::Value token_one_hot;
};

ExpertChoiceRoutingResult ExpertChoiceRoutingKernel(
    spu::SPUContext* spu_ctx, const spu::Value& token_embeddings,
    const spu::Value& gate_weights, const RoutingConfig& config,
    const spu::Value* token_mask = nullptr);

class RoutingServer : public ProtInterface {
 public:
  RoutingServer(size_t token_nums /* A.K.A. m */,
                size_t expert_nums /* A.K.A. n */,
                size_t expert_capacity /* A.K.A. t */,
                size_t embedding_dimension /* A.K.A. d */)
      : config_(token_nums, expert_nums, expert_capacity, embedding_dimension) {
  }

  RoutingServer(const RoutingConfig& config) : config_(config) {}

  // Feed input: token embeddings in shape [m, d]
  // Server provides the actual weight for the Linear layer [d, N]
  void FeedInputs(const std::vector<double>& token_embeddings,
                  const std::vector<double>& gate_weights);

  // Compute the routing: for each expert, select top-t tokens
  void Compute() override;

  // Extract output: dispatch embeddings [N, t, d] and scores [N, t]
  std::vector<double> ExtractOutput();

 private:
  RoutingConfig config_;
};

class RoutingUser : public ProtInterface {
 public:
  RoutingUser(size_t token_nums /* A.K.A. m */,
              size_t expert_nums /* A.K.A. n */,
              size_t expert_capacity /* A.K.A. t */,
              size_t embedding_dimension /* A.K.A. d */)
      : config_(token_nums, expert_nums, expert_capacity, embedding_dimension) {
  }

  RoutingUser(const RoutingConfig& config) : config_(config) {}

  // Feed input: token embeddings in shape [m, d]
  // User provides zero shares for the gate weights (kept private)
  void FeedInputs(const std::vector<double>& token_embeddings);

  // Compute the routing: for each expert, select top-t tokens
  void Compute() override;

  // Extract output: dispatch embeddings [N, t, d] and scores [N, t]
  std::vector<double> ExtractOutput();

 private:
  RoutingConfig config_;
};

// register make MakeProtocol
REGISTER_PROT(Routing);

}  // namespace capricorn
