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

struct CombineConfig {
  // configure parameters
  size_t m_{0};  // token nums
  size_t n_{0};  // expert nums
  size_t t_{0};  // dispatch experts per token
  size_t d_{0};  // embedding dimension

  // variable names in cio
  std::string kExpertsOutEmbeddings{
      "experts_out_embeddings"};  // should be in size n_ * t_ * d_
  std::string kTopTScoresForExperts{
      "top_t_scores_for_experts"};  // should be in size n_ * t_
  std::string kOneHotTokenIndicesForExperts{
      "one_hot_token_indices_for_experts"};  // should be in size n_ * t_ * m_

  // output variable name in cio
  std::string kCombinedEmbeddings{
      "combined_embeddings"};  // should be in size m_ * d_

  CombineConfig(size_t token_nums, size_t expert_nums,
                size_t dispatch_expert_per_token, size_t embedding_dimension)
      : m_(token_nums),
        n_(expert_nums),
        t_(dispatch_expert_per_token),
        d_(embedding_dimension) {}
};

spu::Value CombineKernel(spu::SPUContext* spu_ctx,
                         const spu::Value& experts_out_embeddings,
                         const spu::Value& top_t_scores,
                         const spu::Value& token_one_hot,
                         const CombineConfig& config);

class CombineServer : public ProtInterface {
 public:
  CombineServer(size_t token_nums /* A.K.A. m */,
                size_t expert_nums /* A.K.A. n */,
                size_t dispatch_expert_per_token /* A.K.A. t */,
                size_t embedding_dimension)
      : config_(token_nums, expert_nums, dispatch_expert_per_token,
                embedding_dimension) {}

  CombineServer(const CombineConfig& config) : config_(config) {}

  ~CombineServer() override = default;

  // void InitEnv(std::shared_ptr<yacl::link::Context> lctx) {
  //   env_ptr_ = std::make_unique<Environment>(lctx);
  // }

  // void InitEnv(std::unique_ptr<Environment> env_ptr) {
  //   env_ptr_ = std::move(env_ptr);
  // }

  // FYI, TODO @wenfan
  void FeedInputs(
      const std::vector<double>& experts_out_embeddings,
      const std::vector<double>& top_t_scores_for_experts,
      const std::vector<uint64_t>& one_hot_token_indices_for_experts);

  // compute the dispatch result with SPU
  void Compute() override;

  std::vector<double> ExtractOutput();

  // bool IsEnvReady() { return env_ptr_ != nullptr; }

  // std::unique_ptr<Environment> StealEnv() { return std::move(env_ptr_); }

 private:
  // std::unique_ptr<Environment> env_ptr_{nullptr};
  CombineConfig config_;
};

class CombineUser : public ProtInterface {
 public:
  CombineUser(size_t token_nums /* A.K.A. m */,
              size_t expert_nums /* A.K.A. n */,
              size_t dispatch_expert_per_token /* A.K.A. t */,
              size_t embedding_dimension)
      : config_(token_nums, expert_nums, dispatch_expert_per_token,
                embedding_dimension) {}

  CombineUser(const CombineConfig& config) : config_(config) {}

  ~CombineUser() override = default;

  // void InitEnv(std::shared_ptr<yacl::link::Context> lctx) {
  //   env_ptr_ = std::make_unique<Environment>(lctx);
  // }

  // void InitEnv(std::unique_ptr<Environment> env_ptr) {
  //   env_ptr_ = std::move(env_ptr);
  // }

  // FYI, TODO @wenfan
  void FeedInputs(
      const std::vector<double>& experts_out_embeddings,
      const std::vector<double>& top_t_scores_for_experts,
      const std::vector<uint64_t>& one_hot_token_indices_for_experts);

  // compute the dispatch result with SPU
  void Compute() override;

  std::vector<double> ExtractOutput();

  // bool IsEnvReady() { return env_ptr_ != nullptr; }

  // std::unique_ptr<Environment> StealEnv() { return std::move(env_ptr_); }

 private:
  // std::unique_ptr<Environment> env_ptr_{nullptr};
  CombineConfig config_;
};

// register make MakeProtocol
REGISTER_PROT(Combine);

}  // namespace capricorn
