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

namespace spu::mpc::cheetah {
class BatchMatMul;
}

namespace capricorn {

struct BatchExpertComputeConfig {
  // configure parameters
  size_t n_{0};   // expert nums
  size_t t_{0};   // activated experts per token
  size_t d_{0};   // embedding dimension
  size_t ed_{0};  // expert dimension

  // input variable name in cio
  std::string kSelectedEmbeddingsForExperts{
      "selected_embeddings_for_experts"};  // should be in size n_ * t_ * d_
  std::string kExpertsWeightsOne{
      "experts_weights_one"};  // should be in size n_ * d_ * ed_
  std::string kExpertsWeightsTwo{
      "experts_weights_two"};  // should be in size n_ * ed_ * d_

  // output variable name in cio
  std::string kExpertsOutEmbeddings{
      "experts_out_embeddings"};  // should be in size n_ * t_ * d_

  // Optional benchmark-only persistent sessions.
  // When absent, the original one-session-per-matmul behavior is preserved.
  std::shared_ptr<spu::mpc::cheetah::BatchMatMul> w1_session;
  std::shared_ptr<spu::mpc::cheetah::BatchMatMul> w2_session;

  BatchExpertComputeConfig(size_t expert_nums, size_t token_capacity_per_expert,
                           size_t embedding_dimension, size_t expert_dimension)
      : n_(expert_nums),
        t_(token_capacity_per_expert),
        d_(embedding_dimension),
        ed_(expert_dimension) {}
};

enum class ExpertActivation {
  kSilu,
  kGelu,
};

spu::Value BatchExpertComputeKernel(spu::SPUContext* spu_ctx,
                                    const spu::Value& selected_embeddings,
                                    const spu::Value& experts_weights_one,
                                    const spu::Value& experts_weights_two,
                                    const BatchExpertComputeConfig& config);

spu::Value BatchExpertComputeKernelWithBias(
    spu::SPUContext* spu_ctx, const spu::Value& selected_embeddings,
    const spu::Value& experts_weights_one, const spu::Value& experts_bias_one,
    const spu::Value& experts_weights_two, const spu::Value& experts_bias_two,
    const BatchExpertComputeConfig& config, ExpertActivation activation);

// FYI,
// class FeedForwardExpert:
//     def __init__(d, ed):
//         self.w1 = Linear(d, ed)
//         self.w2 = Linear(ed, d)
//         self.act = GELU / SILU
//     def forward(x):
//         return self.w2(self.act(self.w1(x)))

class BatchExpertComputeServer : public ProtInterface {
 public:
  BatchExpertComputeServer(size_t expert_nums,
                           size_t activated_experts_per_token,
                           size_t embedding_dimension, size_t expert_dimension)
      : config_(expert_nums, activated_experts_per_token, embedding_dimension,
                expert_dimension) {}

  BatchExpertComputeServer(const BatchExpertComputeConfig& config)
      : config_(config) {}

  ~BatchExpertComputeServer() override = default;

  // void InitEnv(std::shared_ptr<yacl::link::Context> lctx) {
  //   env_ptr_ = std::make_unique<Environment>(lctx);
  // }

  // void InitEnv(std::unique_ptr<Environment> env_ptr) {
  //   env_ptr_ = std::move(env_ptr);
  // }

  void FeedInputs(const std::vector<double>& selected_embeddings_for_experts,
                  const std::vector<double>& experts_weights_one,
                  const std::vector<double>& experts_weights_two);

  // compute the dispatch result with SPU
  void Compute() override;

  std::vector<double> ExtractOutput();

  // bool IsEnvReady() { return env_ptr_ != nullptr; }

  // std::unique_ptr<Environment> StealEnv() { return std::move(env_ptr_); }

 private:
  // std::unique_ptr<Environment> env_ptr_{nullptr};
  BatchExpertComputeConfig config_;
};

class BatchExpertComputeUser : public ProtInterface {
 public:
  BatchExpertComputeUser(size_t expert_nums, size_t activated_experts_per_token,
                         size_t embedding_dimension, size_t expert_dimension)
      : config_(expert_nums, activated_experts_per_token, embedding_dimension,
                expert_dimension) {}

  BatchExpertComputeUser(const BatchExpertComputeConfig& config)
      : config_(config) {}

  ~BatchExpertComputeUser() override = default;

  // void InitEnv(std::shared_ptr<yacl::link::Context> lctx) {
  //   env_ptr_ = std::make_unique<Environment>(lctx);
  // }

  // void InitEnv(std::unique_ptr<Environment> env_ptr) {
  //   env_ptr_ = std::move(env_ptr);
  // }

  void FeedInputs(const std::vector<double>& selected_embeddings_for_experts);

  // compute the dispatch result with SPU
  void Compute() override;

  std::vector<double> ExtractOutput();

  // bool IsEnvReady() { return env_ptr_ != nullptr; }

  // std::unique_ptr<Environment> StealEnv() { return std::move(env_ptr_); }

 private:
  // std::unique_ptr<Environment> env_ptr_{nullptr};
  BatchExpertComputeConfig config_;
};

// register make MakeProtocol
REGISTER_PROT(BatchExpertCompute);

}  // namespace capricorn
