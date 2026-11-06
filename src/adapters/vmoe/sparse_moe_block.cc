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

#include "src/adapters/vmoe/sparse_moe_block.h"

#include "libspu/kernel/hal/shape_ops.h"
#include "src/adapters/vmoe/metrics.h"
#include "src/baselines/cryptomoe/dispatch.h"
#include "src/common/moe/combine.h"
#include "src/common/moe/expert_compute_spu.h"
#include "src/common/moe/token_topk.h"

#include "yacl/base/exception.h"

namespace capricorn::vmoe {

spu::Value RunSparseMoeBlock(spu::SPUContext* spu_ctx,
                             const spu::Value& token_embeddings,
                             const spu::Value& router_weights,
                             const spu::Value& experts_weights_one,
                             const spu::Value& experts_weights_two,
                             const SparseMoeConfig& config) {
  YACL_ENFORCE(spu_ctx != nullptr, "SPU context must not be null");
  config.Validate();

  const auto token_count = static_cast<int64_t>(config.gate.m_);
  const auto expert_count = static_cast<int64_t>(config.gate.n_);
  const auto embedding_dim = static_cast<int64_t>(config.gate.d_);
  const auto expert_dim = static_cast<int64_t>(config.expert.ed_);

  YACL_ENFORCE(
      token_embeddings.shape() == spu::Shape({1, token_count, embedding_dim}),
      "token embeddings must have shape [1,{},{}], got {}", token_count,
      embedding_dim, token_embeddings.shape());
  YACL_ENFORCE(
      router_weights.shape() == spu::Shape({embedding_dim, expert_count}),
      "router weights must have official shape [{},{}], got {}", embedding_dim,
      expert_count, router_weights.shape());
  YACL_ENFORCE(experts_weights_one.shape() ==
                   spu::Shape({expert_count, embedding_dim, expert_dim}),
               "expert W1 has invalid shape {}", experts_weights_one.shape());
  YACL_ENFORCE(experts_weights_two.shape() ==
                   spu::Shape({expert_count, expert_dim, embedding_dim}),
               "expert W2 has invalid shape {}", experts_weights_two.shape());

  auto flattened_tokens = spu::kernel::hal::reshape(
      spu_ctx, token_embeddings, {token_count, embedding_dim});
  auto baseline_router_weights =
      spu::kernel::hal::transpose(spu_ctx, router_weights, {1, 0});

  auto routing = GateRoutingKernel(spu_ctx, flattened_tokens,
                                   baseline_router_weights, config.gate);
  auto dispatch =
      DispatchKernel(spu_ctx, flattened_tokens, routing.expert_scores,
                     routing.expert_indices, config.dispatch);
  auto expert_output = BatchExpertComputeKernel(
      spu_ctx, dispatch.selected_embeddings, experts_weights_one,
      experts_weights_two, config.expert);
  auto combined =
      CombineKernel(spu_ctx, expert_output, dispatch.selected_scores,
                    dispatch.token_one_hot, config.combine);

  return spu::kernel::hal::reshape(spu_ctx, combined,
                                   {1, token_count, embedding_dim});
}

spu::Value RunSparseMoeBlockWithBias(
    spu::SPUContext* spu_ctx, const spu::Value& token_embeddings,
    const spu::Value& router_weights, const spu::Value& experts_weights_one,
    const spu::Value& experts_bias_one, const spu::Value& experts_weights_two,
    const spu::Value& experts_bias_two, const SparseMoeConfig& config,
    ExpertActivation activation) {
  YACL_ENFORCE(spu_ctx != nullptr, "SPU context must not be null");
  config.Validate();
  const auto token_count = static_cast<int64_t>(config.gate.m_);
  const auto expert_count = static_cast<int64_t>(config.gate.n_);
  const auto embedding_dim = static_cast<int64_t>(config.gate.d_);
  const auto expert_dim = static_cast<int64_t>(config.expert.ed_);
  YACL_ENFORCE(token_embeddings.shape() ==
               spu::Shape({1, token_count, embedding_dim}));
  YACL_ENFORCE(router_weights.shape() ==
               spu::Shape({embedding_dim, expert_count}));
  YACL_ENFORCE(experts_weights_one.shape() ==
               spu::Shape({expert_count, embedding_dim, expert_dim}));
  YACL_ENFORCE(experts_bias_one.shape() ==
               spu::Shape({expert_count, expert_dim}));
  YACL_ENFORCE(experts_weights_two.shape() ==
               spu::Shape({expert_count, expert_dim, embedding_dim}));
  YACL_ENFORCE(experts_bias_two.shape() ==
               spu::Shape({expert_count, embedding_dim}));

  auto flattened_tokens = spu::kernel::hal::reshape(
      spu_ctx, token_embeddings, {token_count, embedding_dim});
  auto baseline_router_weights =
      spu::kernel::hal::transpose(spu_ctx, router_weights, {1, 0});
  auto routing = [&] {
    StageMetric metric(spu_ctx->lctx(), "moe", "token_choice.gate_routing");
    return GateRoutingKernel(spu_ctx, flattened_tokens, baseline_router_weights,
                             config.gate);
  }();
  auto dispatch = [&] {
    StageMetric metric(spu_ctx->lctx(), "moe", "token_choice.dispatch");
    return DispatchKernel(spu_ctx, flattened_tokens, routing.expert_scores,
                          routing.expert_indices, config.dispatch);
  }();
  auto expert_output = [&] {
    StageMetric metric(spu_ctx->lctx(), "moe",
                       activation == ExpertActivation::kSilu
                           ? "token_choice.expert_silu"
                           : "token_choice.expert_gelu");
    return BatchExpertComputeKernelWithBias(
        spu_ctx, dispatch.selected_embeddings, experts_weights_one,
        experts_bias_one, experts_weights_two, experts_bias_two, config.expert,
        activation);
  }();
  auto combined = [&] {
    StageMetric metric(spu_ctx->lctx(), "moe", "token_choice.combine");
    return CombineKernel(spu_ctx, expert_output, dispatch.selected_scores,
                         dispatch.token_one_hot, config.combine);
  }();
  return spu::kernel::hal::reshape(spu_ctx, combined,
                                   {1, token_count, embedding_dim});
}

}  // namespace capricorn::vmoe
