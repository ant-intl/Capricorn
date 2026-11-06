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

#include "src/capricorn/routing.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "libspu/core/config.h"
#include "libspu/core/encoding.h"
#include "libspu/core/half.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "libspu/device/io.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/basic_unary.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/geometrical.h"
#include "libspu/kernel/hlo/indexing.h"
#include "libspu/kernel/hlo/rank.h"
#include "libspu/kernel/hlo/reduce.h"
#include "libspu/kernel/hlo/shift.h"
#include "libspu/mpc/factory.h"
#include "spdlog/spdlog.h"
#include "src/common/nn/softmax.h"
#include "src/common/runtime/macro.h"
#include "src/common/selection/customized_sort.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/views/xview.hpp"

#include "yacl/base/exception.h"

namespace capricorn {

namespace {

std::vector<spu::Value> PublicBooleanOnehot(spu::SPUContext* ctx,
                                            int64_t one_hot_size) {
  SPDLOG_INFO("PublicBooleanOnehot: one_hot_size {}", one_hot_size);
  SPU_ENFORCE(one_hot_size > 0, "one_hot_size must be positive");

  // Create a range [0, 1, ..., m-1]
  int64_t chunk_num = (one_hot_size + 63) / 64;

  std::vector<std::vector<uint64_t>> range_vec(
      chunk_num, std::vector<uint64_t>(one_hot_size, 0));

  for (int64_t i = 0; i < one_hot_size; ++i) {
    auto chunk_id = i / 64;
    auto chunk_offset = i % 64;
    range_vec[chunk_id][i] = ((uint64_t)1 << chunk_offset);
  }

  // Create mask matrix: [t, m]
  std::vector<spu::Value> onehot_vec;
  for (int64_t i = 0; i < chunk_num; ++i) {
    spu::Value one_hot =
        spu::kernel::hlo::Constant(ctx, range_vec[i], {one_hot_size});
    SPDLOG_INFO("PublicBooleanOnehot: {} one_hot shape {}", i, one_hot.shape());
    onehot_vec.push_back(one_hot);
  }

  return onehot_vec;
}

spu::Value BatchOnehotFromBoolean(spu::SPUContext* ctx,
                                  const std::vector<spu::Value>& bool_inputs,
                                  int64_t one_hot_size) {
  size_t chunk_num = (one_hot_size + 63) / 64;
  SPU_ENFORCE(chunk_num <= bool_inputs.size());

  std::vector<spu::Value> inputs;
  for (size_t i = 0; i < chunk_num; ++i) {
    // SPU_ENFORCE(bool_inputs[i].numel() == one_hot_size);
    inputs.push_back(spu::kernel::hal::reshape(ctx, bool_inputs[i],
                                               {bool_inputs[i].numel(), 1}));
  }

  // auto input =
  //     spu::kernel::hal::reshape(ctx, bool_input, {bool_input.numel(), 1});

  std::vector<spu::Value> shift_vec;
  for (int64_t i = 0; i < one_hot_size; ++i) {
    auto chunk_id = i / 64;
    auto chunk_offset = i % 64;
    shift_vec.push_back(spu::kernel::hal::right_shift_logical(
        ctx, inputs[chunk_id], {chunk_offset}));
  }
  auto shift_matrix = spu::kernel::hlo::Concatenate(ctx, shift_vec, 1);
  auto true_matrix =
      spu::kernel::hlo::Constant(ctx, true, shift_matrix.shape());

  return spu::kernel::hal::bitwise_and(ctx, shift_matrix, true_matrix);
}

}  // namespace

static ExpertChoiceRoutingResult ExpertChoiceRoutingKernelTimed(
    spu::SPUContext* spu_ctx, const spu::Value& token_embeddings,
    const spu::Value& gate_weights, const RoutingConfig& config,
    const spu::Value* token_mask, bool emit_timing) {
  YACL_ENFORCE(spu_ctx != nullptr, "SPU context must not be null");
  YACL_ENFORCE(config.t_ > 0 && config.t_ <= config.m_,
               "expert capacity must be in [1, {}], got {}", config.m_,
               config.t_);
  YACL_ENFORCE(
      token_embeddings.shape() == spu::Shape({static_cast<int64_t>(config.m_),
                                              static_cast<int64_t>(config.d_)}),
      "token embeddings must have shape [{},{}], got {}", config.m_, config.d_,
      token_embeddings.shape());
  YACL_ENFORCE(
      gate_weights.shape() == spu::Shape({static_cast<int64_t>(config.n_),
                                          static_cast<int64_t>(config.d_)}),
      "gate weights must have shape [{},{}], got {}", config.n_, config.d_,
      gate_weights.shape());
  YACL_ENFORCE(token_embeddings.isSecret(), "token embeddings must be secret");
  YACL_ENFORCE(gate_weights.isPrivate(), "gate weights must be private");
  if (token_mask != nullptr) {
    YACL_ENFORCE(
        token_mask->shape() == spu::Shape({static_cast<int64_t>(config.m_)}),
        "token mask must have shape [{}], got {}", config.m_,
        token_mask->shape());
    YACL_ENFORCE(token_mask->isPublic(), "token mask must be public");
  }

  auto start = std::chrono::high_resolution_clock::now();
  if (emit_timing) {
    SPDLOG_INFO("--- ExpertChoiceRoutingKernel START ---");
  }
  SPDLOG_INFO("Config: m={}, d={}, N={}, t={}", config.m_, config.d_, config.n_,
              config.t_);

  const auto& x = token_embeddings;
  SPDLOG_INFO("X shape: {}, W_gate shape: {}", x.shape(), gate_weights.shape());
  // Transpose gate_weights from [n, d] to [d, n]
  auto gate_weights_T =
      spu::kernel::hal::transpose(spu_ctx, gate_weights, {1, 0});
  SPDLOG_INFO("RoutingImpl: gate_weights_T shape {}", gate_weights_T.shape());
  // 2. Compute routing scores: W = Softmax(X @ gate_weights_T) -> [m, n]
  // Note: X @ gate_weights_T where X:[m,d] and gate_weights_T:[d,n] -> [m,n]
  auto logits = spu::kernel::hlo::Dot(spu_ctx, x, gate_weights_T);  // [m, n]
  SPDLOG_INFO("W (before softmax) shape: {}", logits.shape());

  //   // Debug: Print logits before Softmax
  //   auto public_logits = spu::kernel::hal::dump_public_as<double>(
  //       spu_ctx, spu::kernel::hal::reveal(spu_ctx, logits));
  //   SPDLOG_INFO("logits (before softmax) -> {}",
  //   std::vector<double>(public_logits.begin(),
  //                                                                    public_logits.end()));

  auto scores = customized::Softmax_2d(
      spu_ctx, logits, /*axis=*/1);  // softmax over experts dimension
  if (token_mask != nullptr) {
    auto mask_column = spu::kernel::hal::reshape(
        spu_ctx, *token_mask, {static_cast<int64_t>(config.m_), 1});
    auto score_mask =
        spu::kernel::hal::broadcast_to(spu_ctx, mask_column, scores.shape());
    scores = spu::kernel::hal::mul(spu_ctx, scores, score_mask);
  }
  SPDLOG_INFO("scores shape: {}", scores.shape());

  //   // Debug: Print scores after Softmax
  //   auto public_scores = spu::kernel::hal::dump_public_as<double>(
  //       spu_ctx, spu::kernel::hal::reveal(spu_ctx, scores));
  //   SPDLOG_INFO("scores (after softmax) -> {}",
  //   std::vector<double>(public_scores.begin(),
  //                                                                    public_scores.end()));

  auto scores_T = spu::kernel::hal::transpose(spu_ctx, scores, {1, 0});
  auto lctx = spu_ctx->lctx();

  auto topk_outputs = [&] {
    auto onehot_matrix = [&] {
      COMM_TIMER_START(lctx, CAPRICORN_ONE_HOT_TEMPLATE);
      std::vector<spu::Value> result;
      auto onehots = PublicBooleanOnehot(spu_ctx, config.m_);

      for (size_t i = 0; i < onehots.size(); ++i) {
        auto onehot = spu::kernel::hal::reshape(
            spu_ctx, onehots[i], {1, static_cast<int64_t>(config.m_)});
        result.push_back(
            spu::kernel::hal::broadcast_to(spu_ctx, onehot, scores_T.shape()));
      }
      COMM_TIMER_END_PRINT(lctx, CAPRICORN_ONE_HOT_TEMPLATE);
      return result;
    }();

    // auto onehot = spu::kernel::hal::reshape(
    //     spu_ctx, PublicBooleanOnehot(spu_ctx, config.m_),
    //     {1, static_cast<int64_t>(config.m_)});
    // auto onehot_matrix =
    //     spu::kernel::hal::broadcast_to(spu_ctx, onehot, scores_T.shape());

    // auto public_scores_T = spu::kernel::hal::dump_public_as<double>(
    //     spu_ctx, spu::kernel::hal::reveal(spu_ctx, scores_T));
    // SPDLOG_INFO("all score^T -> {}",
    // std::vector<double>(public_scores_T.begin(), public_scores_T.end()));

    //   auto public_onehot_matrix =
    //       spu::kernel::hal::dump_public_as<uint64_t>(spu_ctx, onehot_matrix);
    //   SPDLOG_INFO("onehot matrix-> {}",
    //               std::vector<uint64_t>(public_onehot_matrix.begin(),
    //                                 public_onehot_matrix.end()));
    COMM_TIMER_START(lctx, CAPRICORN_EXPERT_WISE_TOP_T_INCLUSIVE);
    auto values = customized::sort_2d(spu_ctx, scores_T, onehot_matrix);
    auto sorted = values[0];
    std::vector<spu::Value> bool_indices(values.begin() + 1, values.end());
    COMM_TIMER_END_PRINT(lctx, CAPRICORN_EXPERT_WISE_TOP_T_INCLUSIVE);
    return std::make_pair(std::move(sorted), std::move(bool_indices));
  }();
  auto sorted = std::move(topk_outputs.first);
  auto bool_indices = std::move(topk_outputs.second);
  // auto bool_indices = values[1];

  // auto public_bool_indices = spu::kernel::hal::dump_public_as<uint64_t>(
  //     spu_ctx, spu::kernel::hal::reveal(spu_ctx, bool_indices));
  // SPDLOG_INFO("bool indices -> {}",
  //             std::vector<double>(public_bool_indices.begin(),
  //                                 public_bool_indices.end()));

  SPDLOG_INFO("sorted shape: {}", sorted.shape());

  // 3. For each expert, select top-t tokens
  std::vector<spu::Value> selected_embeddings_vec;
  std::vector<spu::Value> selected_token_scores_vec;
  std::vector<spu::Value> selected_token_indices_vec;

  for (size_t i = 0; i < config.n_; ++i) {
    SPDLOG_INFO("Processing expert {}/{}", i + 1, config.n_);
    auto selected_tokens = [&] {
      COMM_TIMER_START(lctx, CAPRICORN_EXPERT_WISE_TOP_T_INCLUSIVE);
      std::vector<spu::Value> cur_bool_indices;
      for (size_t j = 0; j < bool_indices.size(); ++j) {
        cur_bool_indices.push_back(spu::kernel::hal::slice(
            spu_ctx, bool_indices[j], {static_cast<int64_t>(i), 0},
            {static_cast<int64_t>(i + 1), static_cast<int64_t>(config.t_)}));
      }

      // auto cur_bool_index = spu::kernel::hal::slice(
      //     spu_ctx, bool_indices, {static_cast<int64_t>(i), 0},
      //     {static_cast<int64_t>(i + 1), static_cast<int64_t>(config.t_)});
      auto selected_token_scores = spu::kernel::hal::slice(
          spu_ctx, sorted, {static_cast<int64_t>(i), 0},
          {static_cast<int64_t>(i + 1), static_cast<int64_t>(config.t_)});
      COMM_TIMER_END_PRINT(lctx, CAPRICORN_EXPERT_WISE_TOP_T_INCLUSIVE);
      return std::make_pair(std::move(cur_bool_indices),
                            std::move(selected_token_scores));
    }();
    auto cur_bool_indices = std::move(selected_tokens.first);
    auto selected_token_scores = std::move(selected_tokens.second);

    // auto public_scores = spu::kernel::hal::dump_public_as<double>(
    //     spu_ctx, spu::kernel::hal::reveal(spu_ctx, selected_token_scores));
    // SPDLOG_INFO(
    //     "{} - score -> {}", i,
    //     std::vector<double>(public_scores.begin(), public_scores.end()));

    // auto public_onehot = spu::kernel::hal::dump_public_as<uint64_t>(
    //     spu_ctx, spu::kernel::hal::reveal(spu_ctx, batch_onehot));
    // SPDLOG_INFO(
    //     "{} - onehot -> {}", i,
    //     std::vector<uint64_t>(public_onehot.begin(), public_onehot.end()));

    auto batch_onehot = [&] {
      COMM_TIMER_START(lctx, CAPRICORN_ONE_HOT_UNPACK);
      auto result =
          BatchOnehotFromBoolean(spu_ctx, cur_bool_indices, config.m_);
      COMM_TIMER_END_PRINT(lctx, CAPRICORN_ONE_HOT_UNPACK);
      return result;
    }();
    auto selected_embeddings = spu::kernel::hlo::Dot(spu_ctx, batch_onehot, x);
    SPDLOG_INFO("selected_embeddings shape {}", selected_embeddings.shape());

    // cio_ptr->deviceSetVar(fmt::format("selected_embeddings_for_expert_{}",
    // i),
    //                       selected_embeddings);
    spu::Shape shape_3d = selected_embeddings.shape();
    shape_3d.insert(shape_3d.begin(), 1);
    selected_embeddings_vec.push_back(
        spu::kernel::hal::reshape(spu_ctx, selected_embeddings, shape_3d));

    selected_token_scores_vec.push_back(spu::kernel::hal::reshape(
        spu_ctx, selected_token_scores, {1, static_cast<int64_t>(config.t_)}));
    selected_token_indices_vec.push_back(spu::kernel::hal::reshape(
        spu_ctx, batch_onehot,
        {1, static_cast<int64_t>(config.t_), static_cast<int64_t>(config.m_)}));
  }

  auto selected_embeddings_for_experts =
      spu::kernel::hal::concatenate(spu_ctx, selected_embeddings_vec, 0);
  SPDLOG_INFO("selected_embeddings_for_experts shape {}",
              selected_embeddings_for_experts.shape());

  auto selected_token_scores =
      spu::kernel::hal::concatenate(spu_ctx, selected_token_scores_vec, 0);
  SPDLOG_INFO("selected_token_scores shape {}", selected_token_scores.shape());

  auto selected_token_indices =
      spu::kernel::hal::concatenate(spu_ctx, selected_token_indices_vec, 0);
  SPDLOG_INFO("selected_token_indices shape {}",
              selected_token_indices.shape());

  auto end = std::chrono::high_resolution_clock::now();
  auto second =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();
  if (emit_timing) {
    SPDLOG_INFO("--- ExpertChoiceRoutingKernel DONE --- time: {} ms",
                second * 1000);
  }
  return {
      std::move(selected_embeddings_for_experts),
      std::move(selected_token_scores),
      std::move(selected_token_indices),
  };
}

ExpertChoiceRoutingResult ExpertChoiceRoutingKernel(
    spu::SPUContext* spu_ctx, const spu::Value& token_embeddings,
    const spu::Value& gate_weights, const RoutingConfig& config,
    const spu::Value* token_mask) {
  return ExpertChoiceRoutingKernelTimed(spu_ctx, token_embeddings, gate_weights,
                                        config, token_mask,
                                        /*emit_timing=*/true);
}

// RoutingServer implementation
void RoutingServer::FeedInputs(const std::vector<double>& token_embeddings,
                               const std::vector<double>& gate_weights) {
  SPDLOG_INFO("RoutingServer::FeedInputs");
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");

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
  auto xt_token_embeddings = xt::adapt(
      token_embeddings_copy, std::vector<size_t>{token_nums, embedding_size});
  auto xt_gate_weights = xt::adapt(
      gate_weights_copy, std::vector<size_t>{expert_nums, embedding_size});

  env_ptr_->FeedShare(config_.kTokenEmbeddings, xt_token_embeddings);
  env_ptr_->SetPrivate(config_.kGateWeights, xt_gate_weights);

  SPDLOG_INFO("RoutingServer FeedInputs DONE");
}

void RoutingServer::Compute() {
  SPDLOG_INFO("RoutingServer::Compute");
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  auto start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("--- ExpertChoiceRoutingKernel START ---");
  auto result = ExpertChoiceRoutingKernelTimed(
      env_ptr_->spu_ctx_.get(),
      env_ptr_->cio_->deviceGetVar(config_.kTokenEmbeddings),
      env_ptr_->cio_->deviceGetVar(config_.kGateWeights), config_, nullptr,
      /*emit_timing=*/false);
  env_ptr_->cio_->deviceSetVar(config_.kSelectedEmbeddingsForExperts,
                               std::move(result.selected_embeddings));
  env_ptr_->cio_->deviceSetVar(config_.kTopTScoresForExperts,
                               std::move(result.selected_scores));
  env_ptr_->cio_->deviceSetVar(config_.kOneHotTokenIndicesForExperts,
                               std::move(result.token_one_hot));
  auto end = std::chrono::high_resolution_clock::now();
  auto second =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();
  SPDLOG_INFO("--- ExpertChoiceRoutingKernel DONE --- time: {} ms",
              second * 1000);
}

std::vector<double> RoutingServer::ExtractOutput() {
  SPDLOG_INFO("RoutingServer::ExtractOutput");
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  auto xt_selected_embeddings = spu::kernel::hal::dump_public_as<double>(
      env_ptr_->spu_ctx_.get(),
      spu::kernel::hal::reveal(
          env_ptr_->spu_ctx_.get(),
          env_ptr_->cio_->deviceGetVar(config_.kSelectedEmbeddingsForExperts)));

  std::vector<double> selected_embeddings(xt_selected_embeddings.begin(),
                                          xt_selected_embeddings.end());
  return selected_embeddings;
}

// RoutingUser implementation
void RoutingUser::FeedInputs(const std::vector<double>& token_embeddings) {
  SPDLOG_INFO("RoutingUser::FeedInputs");
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");

  auto token_nums = config_.m_;
  auto embedding_size = config_.d_;

  // Validate input size
  YACL_ENFORCE(token_embeddings.size() == token_nums * embedding_size,
               "Invalid token_embeddings size");

  // Make copy for xt::adapt
  auto token_embeddings_copy = token_embeddings;

  // Adapt to xtensor array
  auto xt_token_embeddings = xt::adapt(
      token_embeddings_copy, std::vector<size_t>{token_nums, embedding_size});

  env_ptr_->FeedShare(config_.kTokenEmbeddings, xt_token_embeddings);
  env_ptr_->GetPrivate(config_.kGateWeights);

  SPDLOG_INFO("RoutingUser FeedInputs DONE");
}

void RoutingUser::Compute() {
  SPDLOG_INFO("RoutingUser::Compute");
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  auto start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("--- ExpertChoiceRoutingKernel START ---");
  auto result = ExpertChoiceRoutingKernelTimed(
      env_ptr_->spu_ctx_.get(),
      env_ptr_->cio_->deviceGetVar(config_.kTokenEmbeddings),
      env_ptr_->cio_->deviceGetVar(config_.kGateWeights), config_, nullptr,
      /*emit_timing=*/false);
  env_ptr_->cio_->deviceSetVar(config_.kSelectedEmbeddingsForExperts,
                               std::move(result.selected_embeddings));
  env_ptr_->cio_->deviceSetVar(config_.kTopTScoresForExperts,
                               std::move(result.selected_scores));
  env_ptr_->cio_->deviceSetVar(config_.kOneHotTokenIndicesForExperts,
                               std::move(result.token_one_hot));
  auto end = std::chrono::high_resolution_clock::now();
  auto second =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();
  SPDLOG_INFO("--- ExpertChoiceRoutingKernel DONE --- time: {} ms",
              second * 1000);
}

std::vector<double> RoutingUser::ExtractOutput() {
  SPDLOG_INFO("RoutingUser::ExtractOutput");
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");

  auto xt_selected_embeddings = spu::kernel::hal::dump_public_as<double>(
      env_ptr_->spu_ctx_.get(),
      spu::kernel::hal::reveal(
          env_ptr_->spu_ctx_.get(),
          env_ptr_->cio_->deviceGetVar(config_.kSelectedEmbeddingsForExperts)));

  std::vector<double> selected_embeddings(xt_selected_embeddings.begin(),
                                          xt_selected_embeddings.end());
  return selected_embeddings;
}

}  // namespace capricorn
