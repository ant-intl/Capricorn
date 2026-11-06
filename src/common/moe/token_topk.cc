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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "libspu/core/config.h"
#include "libspu/core/encoding.h"
#include "libspu/core/half.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "libspu/device/io.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/basic_ternary.h"
#include "libspu/kernel/hlo/basic_unary.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/geometrical.h"
#include "libspu/kernel/hlo/indexing.h"
#include "libspu/kernel/hlo/rank.h"
#include "libspu/kernel/hlo/reduce.h"
#include "libspu/mpc/factory.h"

// Include for hal::add, hal::max
#include "libspu/kernel/hal/polymorphic.h"
#include "spdlog/spdlog.h"
#include "src/common/nn/softmax.h"
#include "src/common/runtime/macro.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"

namespace capricorn {

// Gate Routing Implementation
// Computes: W, K = Top-K(Softmax(Linear(x)), k)
// where Linear(x) = x @ W^T, W is [n, d] gate weight matrix
static GateRoutingResult GateRoutingKernelTimed(spu::SPUContext* spu_ctx,
                                                const spu::Value& x,
                                                const spu::Value& gate_weights,
                                                const GateRoutingConfig& config,
                                                bool emit_timing) {
  auto start = std::chrono::high_resolution_clock::now();
  if (emit_timing) {
    SPDLOG_INFO("--- GateRouting START ---");
  }

  SPDLOG_INFO("GateRoutingImpl: x shape {}", x.shape());
  SPDLOG_INFO("GateRoutingImpl: gate_weights shape {}", gate_weights.shape());

  YACL_ENFORCE(x.shape() == spu::Shape({static_cast<int64_t>(config.m_),
                                        static_cast<int64_t>(config.d_)}));
  YACL_ENFORCE(gate_weights.shape() ==
               spu::Shape({static_cast<int64_t>(config.n_),
                           static_cast<int64_t>(config.d_)}));

  // Step 1: Linear transformation - compute logits = x @ W^T
  // x: [m, d], W: [n, d] -> W^T: [d, n] -> logits: [m, n]
  SPDLOG_INFO("GateRoutingImpl: computing Linear(x) = x @ W^T");
  // Transpose gate_weights from [n, d] to [d, n]
  auto gate_weights_T =
      spu::kernel::hal::transpose(spu_ctx, gate_weights, {1, 0});
  SPDLOG_INFO("GateRoutingImpl: gate_weights_T shape {}",
              gate_weights_T.shape());
  // Compute x @ W^T
  auto logits = spu::kernel::hlo::Dot(spu_ctx, x, gate_weights_T);
  SPDLOG_INFO("GateRoutingImpl: logits shape {}", logits.shape());

  // Step 2: Softmax over expert dimension (dim=-1, which is axis=1 for [m, n])
  // G(x) = Softmax(logits) -> [m, n]
  SPDLOG_INFO("GateRoutingImpl: computing Softmax");
  auto scores = customized::Softmax_2d(spu_ctx, logits, /*axis=*/1);
  SPDLOG_INFO("GateRoutingImpl: scores shape {}", scores.shape());

  // Step 3: Top-K selection
  // W, K = Top-K(scores, k) -> W: [m, k], K: [m, k]
  SPDLOG_INFO("GateRoutingImpl: computing Top-K with k={}", config.k_);
  auto lctx = spu_ctx->lctx();
  COMM_TIMER_START(lctx, CRYPTOMOE_TOKEN_WISE_TOPK);
  auto topk_result = spu::kernel::hlo::TopK(spu_ctx, scores, config.k_);
  // Fixed-point Exp/Div can overshoot the mathematical Softmax range by a few
  // ulps (for example, 1.000015 at f=18). Dispatch uses M + W as its ranking
  // key, so enforce the Softmax interface contract on only the m*k selected
  // scores. This required post-processing remains part of token-wise Top-k's
  // measured cost.
  auto score_min = spu::kernel::hal::zeros(
      spu_ctx, topk_result[0].dtype(), topk_result[0].shape());
  auto score_max = spu::kernel::hal::constant(
      spu_ctx, 1.0F, topk_result[0].dtype(), topk_result[0].shape());
  topk_result[0] = spu::kernel::hlo::Clamp(
      spu_ctx, topk_result[0], score_min, score_max);
  COMM_TIMER_END_PRINT(lctx, CRYPTOMOE_TOKEN_WISE_TOPK);
  // topk_result[0] = values (scores), topk_result[1] = indices
  SPDLOG_INFO("GateRoutingImpl: Top-K done");

  auto end = std::chrono::high_resolution_clock::now();
  auto second =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();

  if (emit_timing) {
    SPDLOG_INFO("--- GateRouting DONE --- time: {} ms", second * 1000);
  }
  return {topk_result[0], topk_result[1]};
}

GateRoutingResult GateRoutingKernel(spu::SPUContext* spu_ctx,
                                    const spu::Value& x,
                                    const spu::Value& gate_weights,
                                    const GateRoutingConfig& config) {
  return GateRoutingKernelTimed(spu_ctx, x, gate_weights, config,
                                /*emit_timing=*/true);
}

namespace {

void GateRoutingImpl(spu::SPUContext* spu_ctx,
                     spu::device::ColocatedIo* cio_ptr,
                     const GateRoutingConfig& config) {
  auto start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("--- GateRouting START ---");

  auto result = GateRoutingKernelTimed(
      spu_ctx, cio_ptr->deviceGetVar(config.kTokenEmbeddings),
      cio_ptr->deviceGetVar(config.kGateWeights), config,
      /*emit_timing=*/false);
  cio_ptr->deviceSetVar(config.kExpertScores, result.expert_scores);
  cio_ptr->deviceSetVar(config.kExpertIndices, result.expert_indices);
  auto end = std::chrono::high_resolution_clock::now();
  auto second =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();
  SPDLOG_INFO("--- GateRouting DONE --- time: {} ms", second * 1000);
}

// ========== PLAINTEXT VERSION ==========

std::pair<std::vector<double>, std::vector<uint64_t>> PlaintextGateRouting(
    const GateRoutingConfig& config, const xt::xarray<double>& token_embeddings,
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

  // Step 3: Top-K selection
  std::vector<double> top_k_scores(config.m_ * config.k_, 0.0);
  std::vector<uint64_t> top_k_indices(config.m_ * config.k_, 0);

  for (size_t i = 0; i < config.m_; ++i) {
    // Create vector of (score, index) pairs for this token
    std::vector<std::pair<double, uint64_t>> expert_scores;
    for (size_t j = 0; j < config.n_; ++j) {
      expert_scores.push_back({scores(i, j), j});
    }

    // Sort by score descending
    std::sort(expert_scores.begin(), expert_scores.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Take top-k
    for (size_t k = 0; k < config.k_; ++k) {
      top_k_scores[i * config.k_ + k] = expert_scores[k].first;
      top_k_indices[i * config.k_ + k] = expert_scores[k].second;
    }
  }

  SPDLOG_INFO("GateRoutingTest: top_k_scores {}", top_k_scores);

  return {top_k_scores, top_k_indices};
}

std::pair<std::vector<double>, std::vector<uint64_t>> _PlainGateRoutingImpl(
    spu::SPUContext* spu_ctx, spu::device::ColocatedIo* cio_ptr,
    const GateRoutingConfig& config) {
  auto xt_token_embeddings = spu::kernel::hal::dump_public_as<double>(
      spu_ctx, spu::kernel::hal::reveal(
                   spu_ctx, cio_ptr->deviceGetVar(config.kTokenEmbeddings)));

  // Extract expert indices [m, k]
  auto xt_gate_weights = spu::kernel::hal::dump_public_as<double>(
      spu_ctx, spu::kernel::hal::reveal(
                   spu_ctx, cio_ptr->deviceGetVar(config.kGateWeights)));

  return PlaintextGateRouting(config, xt_token_embeddings, xt_gate_weights);
}
}  // namespace

// =====================================================================================================================

void GateRouting(spu::SPUContext* spu_ctx, spu::device::ColocatedIo* cio_ptr,
                 const GateRoutingConfig& config) {
  GateRoutingImpl(spu_ctx, cio_ptr, config);
}

void GateRoutingServer::FeedInputs(const std::vector<double>& token_embeddings,
                                   const std::vector<double>& gate_weights) {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  SPDLOG_INFO("GateRoutingServer FeedInputs");

  auto token_nums = config_.m_;
  auto embedding_size = config_.d_;
  auto expert_nums = config_.n_;

  // Validate input sizes
  YACL_ENFORCE(token_embeddings.size() == token_nums * embedding_size,
               "Invalid token_embeddings size");
  YACL_ENFORCE(gate_weights.size() == expert_nums * embedding_size,
               "Invalid gate_weights size");

  // Make copies for xt::adapt
  auto token_embeddings_copy = token_embeddings;
  auto gate_weights_copy = gate_weights;

  // Adapt to xtensor arrays
  auto xt_token_embeddings =
      xt::adapt(token_embeddings_copy, {token_nums, embedding_size});
  auto xt_gate_weights =
      xt::adapt(gate_weights_copy, {expert_nums, embedding_size});

  env_ptr_->FeedShare(config_.kTokenEmbeddings, xt_token_embeddings);
  env_ptr_->SetPrivate(config_.kGateWeights, xt_gate_weights);

  SPDLOG_INFO("GateRoutingServer SYNC DONE");
}

void GateRoutingServer::Compute() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  GateRouting(env_ptr_->spu_ctx_.get(), env_ptr_->cio_.get(), config_);
}

std::pair<std::vector<double>, std::vector<uint64_t>>
GateRoutingServer::_ExtractExpectedOutput() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  return _PlainGateRoutingImpl(env_ptr_->spu_ctx_.get(), env_ptr_->cio_.get(),
                               config_);
}

std::pair<std::vector<double>, std::vector<uint64_t>>
GateRoutingServer::ExtractOutput() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");

  // Extract expert scores [m, k]
  auto xt_expert_scores = spu::kernel::hal::dump_public_as<double>(
      env_ptr_->spu_ctx_.get(),
      spu::kernel::hal::reveal(
          env_ptr_->spu_ctx_.get(),
          env_ptr_->cio_->deviceGetVar(config_.kExpertScores)));

  // Extract expert indices [m, k]
  auto xt_expert_indices = spu::kernel::hal::dump_public_as<uint64_t>(
      env_ptr_->spu_ctx_.get(),
      spu::kernel::hal::reveal(
          env_ptr_->spu_ctx_.get(),
          env_ptr_->cio_->deviceGetVar(config_.kExpertIndices)));

  std::vector<double> expert_scores(xt_expert_scores.begin(),
                                    xt_expert_scores.end());
  std::vector<uint64_t> expert_indices(xt_expert_indices.begin(),
                                       xt_expert_indices.end());

  return {expert_scores, expert_indices};
}

// =====================================================================================================================

void GateRoutingUser::FeedInputs(const std::vector<double>& token_embeddings) {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  SPDLOG_INFO("GateRoutingUser FeedInputs");

  auto token_nums = config_.m_;
  auto embedding_size = config_.d_;
  // auto expert_nums = config_.n_;

  // Validate input size
  YACL_ENFORCE(token_embeddings.size() == token_nums * embedding_size,
               "Invalid token_embeddings size");

  // Make copy for xt::adapt
  auto token_embeddings_copy = token_embeddings;

  // Adapt to xtensor arrays
  auto xt_token_embeddings =
      xt::adapt(token_embeddings_copy, {token_nums, embedding_size});

  env_ptr_->FeedShare(config_.kTokenEmbeddings, xt_token_embeddings);
  env_ptr_->GetPrivate(config_.kGateWeights);

  SPDLOG_INFO("GateRoutingUser SYNC DONE");
}

void GateRoutingUser::Compute() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  GateRouting(env_ptr_->spu_ctx_.get(), env_ptr_->cio_.get(), config_);
}

std::pair<std::vector<double>, std::vector<uint64_t>>
GateRoutingUser::_ExtractExpectedOutput() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  return _PlainGateRoutingImpl(env_ptr_->spu_ctx_.get(), env_ptr_->cio_.get(),
                               config_);
}

std::pair<std::vector<double>, std::vector<uint64_t>>
GateRoutingUser::ExtractOutput() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");

  // Extract expert scores [m, k]
  auto xt_expert_scores = spu::kernel::hal::dump_public_as<double>(
      env_ptr_->spu_ctx_.get(),
      spu::kernel::hal::reveal(
          env_ptr_->spu_ctx_.get(),
          env_ptr_->cio_->deviceGetVar(config_.kExpertScores)));

  // Extract expert indices [m, k]
  auto xt_expert_indices = spu::kernel::hal::dump_public_as<uint64_t>(
      env_ptr_->spu_ctx_.get(),
      spu::kernel::hal::reveal(
          env_ptr_->spu_ctx_.get(),
          env_ptr_->cio_->deviceGetVar(config_.kExpertIndices)));

  std::vector<double> expert_scores(xt_expert_scores.begin(),
                                    xt_expert_scores.end());
  std::vector<uint64_t> expert_indices(xt_expert_indices.begin(),
                                       xt_expert_indices.end());

  return {expert_scores, expert_indices};
}

}  // namespace capricorn
