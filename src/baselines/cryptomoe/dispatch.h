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
#include "libspu/kernel/hal/prot_wrapper.h"   // bad reference
#include "libspu/kernel/hal/public_helper.h"  // bad reference
#include "src/common/runtime/environment.h"
#include "src/common/runtime/prot_helper.h"
#include "src/common/runtime/prot_interface.h"

namespace capricorn {

enum class ExpertTopTMode {
  SerialQuickSelect,
  ParallelQuickSelect,
};

ExpertTopTMode ParseExpertTopTMode(const std::string& mode);
const char* ExpertTopTModeName(ExpertTopTMode mode);

struct DispatchConfig {
  // configure parameters
  size_t m_{0};  // token nums
  size_t n_{0};  // expert nums
  size_t k_{0};  // activated experts per token
  size_t t_{0};  // dispatch experts per token
  size_t d_{0};  // embedding dimension
  ExpertTopTMode top_t_mode_{ExpertTopTMode::SerialQuickSelect};

  // variable names in cio
  std::string kTokenEmbeddings{
      "token_embeddings"};                       // should be in size m_ * d_
  std::string kExpertIndices{"expert_indices"};  // should be in size k_ * m_
  std::string kExpertScores{"expert_scores"};    // should be in size k_ * m_

  // output variable name in cio
  std::string kSelectedEmbeddingsForExperts{
      "selected_embeddings_for_experts"};  // should be in size n_ * t_ * d_
  std::string kTopTScoresForExperts{
      "top_t_scores_for_experts"};  // should be in size n_ * t_
  std::string kOneHotTokenIndicesForExperts{
      "one_hot_token_indices_for_experts"};  // should be in size n_ * t_ * m_

  DispatchConfig(size_t token_nums, size_t expert_nums,
                 size_t activated_experts_per_token,
                 size_t dispatch_expert_per_token, size_t embedding_dimension,
                 ExpertTopTMode top_t_mode = ExpertTopTMode::SerialQuickSelect)
      : m_(token_nums),
        n_(expert_nums),
        k_(activated_experts_per_token),
        t_(dispatch_expert_per_token),
        d_(embedding_dimension),
        top_t_mode_(top_t_mode) {}
};

struct DispatchResult {
  spu::Value selected_embeddings;
  spu::Value selected_scores;
  spu::Value token_one_hot;
};

DispatchResult DispatchKernel(spu::SPUContext* spu_ctx,
                              const spu::Value& token_embeddings,
                              const spu::Value& expert_scores,
                              const spu::Value& expert_indices,
                              const DispatchConfig& config);

class DispatchServer : public ProtInterface {
 public:
  DispatchServer(size_t token_nums /* A.K.A. m */,
                 size_t expert_nums /* A.K.A. n */,
                 size_t activated_experts_per_token /* A.K.A. k */,
                 size_t dispatch_expert_per_token /* A.K.A. t */,
                 size_t embedding_dimension /* A.K.A. d */)
      : config_(token_nums, expert_nums, activated_experts_per_token,
                dispatch_expert_per_token, embedding_dimension) {}

  DispatchServer(const DispatchConfig& config) : config_(config) {}

  ~DispatchServer() override = default;

  // void InitEnv(std::shared_ptr<yacl::link::Context> lctx) {
  //   env_ptr_ = std::make_unique<Environment>(lctx);
  // }

  // void InitEnv(std::unique_ptr<Environment> env_ptr) {
  //   env_ptr_ = std::move(env_ptr);
  // }

  // FYI,
  // token embeddings should be in the shape of m * d
  // flatten_expert_indices and flatten_expert_scores should be in the
  // shape of k * m
  void FeedInputs(
      const std::vector<double>& token_embeddings /* A.K.A. x */,
      const std::vector<uint64_t>& flatten_expert_indices /* A.K.A. K */,
      const std::vector<double>& flatten_expert_scores /* A.K.A. W */);

  // compute the dispatch result with SPU
  void Compute() override;

  std::vector<double> ExtractOutput();

  // bool IsEnvReady() { return env_ptr_ != nullptr; }

  // std::unique_ptr<Environment> StealEnv() { return std::move(env_ptr_); }

 private:
  // std::unique_ptr<Environment> env_ptr_{nullptr};
  DispatchConfig config_;
};

class DispatchUser : public ProtInterface {
 public:
  DispatchUser(size_t token_nums /* A.K.A. m */,
               size_t expert_nums /* A.K.A. n */,
               size_t activated_experts_per_token /* A.K.A. k */,
               size_t dispatch_expert_per_token /* A.K.A. t */,
               size_t embedding_dimension /* A.K.A. d */)
      : config_(token_nums, expert_nums, activated_experts_per_token,
                dispatch_expert_per_token, embedding_dimension) {}

  DispatchUser(const DispatchConfig& config) : config_(config) {}

  ~DispatchUser() override = default;

  // void InitEnv(std::shared_ptr<yacl::link::Context> lctx) {
  //   env_ptr_ = std::make_unique<Environment>(lctx);
  // }

  // void InitEnv(std::unique_ptr<Environment> env_ptr) {
  //   env_ptr_ = std::move(env_ptr);
  // }

  // FYI,
  // token embeddings should be in the shape of m * d
  // flatten_expert_indices and flatten_expert_scores should be in the
  // shape of k * m
  void FeedInputs(
      const std::vector<double>& token_embeddings /* A.K.A. x */,
      const std::vector<uint64_t>& flatten_expert_indices /* A.K.A. K */,
      const std::vector<double>& flatten_expert_scores /* A.K.A. W */);

  // compute the dispatch result with SPU
  void Compute() override;

  std::vector<double> ExtractOutput();

  // bool IsEnvReady() { return env_ptr_ != nullptr; }

  // std::unique_ptr<Environment> StealEnv() { return std::move(env_ptr_); }

 private:
  // std::unique_ptr<Environment> env_ptr_{nullptr};
  DispatchConfig config_;
};

// register make MakeProtocol
REGISTER_PROT(Dispatch);

}  // namespace capricorn
