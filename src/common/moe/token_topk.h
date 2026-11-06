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
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/prot_helper.h"
#include "src/common/runtime/prot_interface.h"

namespace capricorn {

// Gate Routing Configuration
// Computes: W, K = Top-K(G(x), k), where G(x) = Softmax(Linear(x))
// - x: token embeddings [m, d]
// - W: top-k expert scores [m, k]
// - K: top-k expert indices [m, k]
struct GateRoutingConfig {
  // configure parameters
  size_t m_{0};  // token nums
  size_t n_{0};  // expert nums
  size_t k_{0};  // top-k experts to select
  size_t d_{0};  // embedding dimension

  // variable names in cio
  std::string kTokenEmbeddings{
      "token_embeddings"};  // should be in size m_ * d_
  std::string kGateWeights{"gate_weights"};

  // output variable names in cio
  std::string kExpertScores{
      "expert_scores"};  // should be in size m_ * k_ (top-k scores W)
  std::string kExpertIndices{
      "expert_indices"};  // should be in size m_ * k_ (top-k indices K)

  GateRoutingConfig(size_t token_nums, size_t expert_nums, size_t top_k,
                    size_t embedding_dimension)
      : m_(token_nums), n_(expert_nums), k_(top_k), d_(embedding_dimension) {}
};

struct GateRoutingResult {
  spu::Value expert_scores;
  spu::Value expert_indices;
};

GateRoutingResult GateRoutingKernel(spu::SPUContext* spu_ctx,
                                    const spu::Value& token_embeddings,
                                    const spu::Value& gate_weights,
                                    const GateRoutingConfig& config);

// Reusable implementation entry point. This lets composite protocols include
// GateRouting in a larger synchronized benchmark stage without transferring
// ownership of the Environment between protocol wrappers.
void GateRouting(spu::SPUContext* spu_ctx, spu::device::ColocatedIo* cio_ptr,
                 const GateRoutingConfig& config);

class GateRoutingServer : public ProtInterface {
 public:
  GateRoutingServer(size_t token_nums /* A.K.A. m */,
                    size_t expert_nums /* A.K.A. n */,
                    size_t top_k /* A.K.A. k */,
                    size_t embedding_dimension /* A.K.A. d */)
      : config_(token_nums, expert_nums, top_k, embedding_dimension) {}

  GateRoutingServer(const GateRoutingConfig& config) : config_(config) {}

  ~GateRoutingServer() override = default;

  // Feed input: token embeddings in shape [m, d]
  // Server provides the actual weight for the Linear layer
  void FeedInputs(const std::vector<double>& token_embeddings,
                  const std::vector<double>& gate_weights);

  // Compute the gate routing: W, K = Top-K(Softmax(Linear(x)), k)
  void Compute() override;

  std::pair<std::vector<double>, std::vector<uint64_t>>
  _ExtractExpectedOutput();

  // Extract output: expert scores [m, k] and indices [m, k]
  std::pair<std::vector<double>, std::vector<uint64_t>> ExtractOutput();

 private:
  GateRoutingConfig config_;
};

class GateRoutingUser : public ProtInterface {
 public:
  GateRoutingUser(size_t token_nums /* A.K.A. m */,
                  size_t expert_nums /* A.K.A. n */,
                  size_t top_k /* A.K.A. k */,
                  size_t embedding_dimension /* A.K.A. d */)
      : config_(token_nums, expert_nums, top_k, embedding_dimension) {}

  GateRoutingUser(const GateRoutingConfig& config) : config_(config) {}

  ~GateRoutingUser() override = default;

  // Feed input: token embeddings in shape [m, d]
  // User provides zero shares for the gate weights (kept private)
  void FeedInputs(const std::vector<double>& token_embeddings);

  // Compute the gate routing: W, K = Top-K(Softmax(Linear(x)), k)
  void Compute() override;

  std::pair<std::vector<double>, std::vector<uint64_t>>
  _ExtractExpectedOutput();

  // Extract output: expert scores [m, k] and indices [m, k]
  std::pair<std::vector<double>, std::vector<uint64_t>> ExtractOutput();

 private:
  GateRoutingConfig config_;
};

// register make MakeProtocol
REGISTER_PROT(GateRouting);

}  // namespace capricorn
