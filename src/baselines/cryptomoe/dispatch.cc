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
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/common/runtime/macro.h"
#include "src/common/selection/customized_sort.h"
#include "libspu/core/config.h"
#include "libspu/core/encoding.h"
#include "libspu/core/half.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "libspu/device/io.h"
#include "libspu/kernel/hal/constants.h"  // bad reference
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
#include "libspu/mpc/factory.h"
#include "spdlog/spdlog.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xcsv.hpp"
#include "xtensor/views/xview.hpp"

namespace capricorn {

ExpertTopTMode ParseExpertTopTMode(const std::string& mode) {
  if (mode == "serial") {
    return ExpertTopTMode::SerialQuickSelect;
  }
  if (mode == "parallel") {
    return ExpertTopTMode::ParallelQuickSelect;
  }
  YACL_ENFORCE(false,
               "unsupported expert-wise Top-t mode '{}'; expected serial or "
               "parallel",
               mode);
  return ExpertTopTMode::SerialQuickSelect;
}

const char* ExpertTopTModeName(ExpertTopTMode mode) {
  switch (mode) {
    case ExpertTopTMode::SerialQuickSelect:
      return "serial";
    case ExpertTopTMode::ParallelQuickSelect:
      return "parallel";
  }
  YACL_ENFORCE(false, "unknown expert-wise Top-t mode");
  return "unknown";
}

namespace {

spu::Value BatchOneHot(spu::SPUContext* spu_ctx, const spu::Value& inputs,
                       int64_t onehot_size) {
  std::vector<uint32_t> _vec(onehot_size, 0);
  for (size_t i = 0; i < _vec.size(); ++i) {
    _vec[i] = i;
  }

  spu::Value mask_matrix =
      spu::kernel::hlo::Constant(spu_ctx, _vec, {inputs.numel(), onehot_size});

  spu::Value _inputs =
      spu::kernel::hal::reshape(spu_ctx, inputs, {inputs.numel(), 1});
  spu::Value ext_inputs = spu::kernel::hal::broadcast_to(
      spu_ctx, _inputs, {inputs.numel(), onehot_size});

  return spu::kernel::hlo::Equal(spu_ctx, ext_inputs, mask_matrix);
}

}  // namespace

static DispatchResult DispatchKernelTimed(spu::SPUContext* spu_ctx,
                                          const spu::Value& x,
                                          const spu::Value& expert_scores,
                                          const spu::Value& expert_indices,
                                          const DispatchConfig& config,
                                          bool emit_timing) {
  auto start = std::chrono::high_resolution_clock::now();
  auto lctx = spu_ctx->lctx();
  if (emit_timing) {
    SPDLOG_INFO("--- DispatchImpl START ---");
  }

  auto K = expert_indices;
  auto W = expert_scores;

  YACL_ENFORCE(x.shape() == spu::Shape({static_cast<int64_t>(config.m_),
                                        static_cast<int64_t>(config.d_)}));
  YACL_ENFORCE(K.shape() == spu::Shape({static_cast<int64_t>(config.m_),
                                        static_cast<int64_t>(config.k_)}));
  YACL_ENFORCE(W.shape() == K.shape());

  K = spu::kernel::hal::reshape(spu_ctx, K, {K.numel()});  // flatten
  W = spu::kernel::hal::reshape(spu_ctx, W, {W.numel()});  // flatten

  int64_t select_expert_nums = config.t_;
  int64_t token_nums = config.m_;
  int64_t active_expert_nums = config.k_;

  std::vector<spu::Value> selected_embeddings_vec;
  std::vector<spu::Value> selected_token_scores_vec;
  std::vector<spu::Value> selected_token_indices_vec;

  std::vector<spu::Value> masked_scores_vec;
  std::vector<spu::Value> priority_scores_vec;
  masked_scores_vec.reserve(config.n_);
  priority_scores_vec.reserve(config.n_);
  for (size_t i = 0; i < config.n_; ++i) {
    auto scores = [&] {
      COMM_TIMER_START(lctx, CRYPTOMOE_TOKEN_PRIORITY_SCORES);
      auto current_indices =
          spu::kernel::hlo::Constant(spu_ctx, uint64_t(i), K.shape());
      auto boolean_masks = spu::kernel::hlo::Equal(spu_ctx, K, current_indices);
      // Reproduce CryptoMoE's explicit Step 1 score path:
      //   S = M * W is materialized for every expert/candidate pair and is the
      //   routing weight consumed by Combine.
      //   R = M + W is an independent ranking key used only by Top-t.  With
      //   0 <= W <= 1, assigned candidates rank above unassigned candidates
      //   without introducing the large all-zero partition of S.
      auto masked_scores = spu::kernel::hlo::Mul(spu_ctx, boolean_masks, W);
      auto priority_scores = spu::kernel::hlo::Add(spu_ctx, boolean_masks, W);
      COMM_TIMER_END_PRINT(lctx, CRYPTOMOE_TOKEN_PRIORITY_SCORES);
      return std::make_pair(std::move(masked_scores),
                            std::move(priority_scores));
    }();

    masked_scores_vec.push_back(spu::kernel::hal::reshape(
        spu_ctx, scores.first, {1, scores.first.numel()}));
    priority_scores_vec.push_back(spu::kernel::hal::reshape(
        spu_ctx, scores.second, {1, scores.second.numel()}));
  }

  auto selected_tokens = [&] {
    COMM_TIMER_START(lctx, CRYPTOMOE_EXPERT_WISE_TOP_T);
    SPU_ENFORCE(
        spu_ctx->hasKernel("rand_perm_m") && spu_ctx->hasKernel("perm_am"),
        "Cheetah native permutation kernels are required; refusing "
        "SPU TopK/Shuffle's generic-sort fallback");

    std::vector<spu::Value> ranking;
    if (config.top_t_mode_ == ExpertTopTMode::SerialQuickSelect) {
      SPDLOG_INFO(
          "CryptoMoE expert-wise Top-t mode=serial: calling the "
          "payload-carrying QuickSelect once per expert");
      std::vector<spu::Value> score_rows;
      std::vector<spu::Value> index_rows;
      std::vector<spu::Value> masked_score_rows;
      score_rows.reserve(config.n_);
      index_rows.reserve(config.n_);
      masked_score_rows.reserve(config.n_);
      for (size_t i = 0; i < config.n_; ++i) {
        auto row_ranking = customized::batched_quick_select_topk_2d(
            spu_ctx, priority_scores_vec[i], {masked_scores_vec[i]},
            select_expert_nums);
        SPU_ENFORCE(row_ranking.size() == 3);
        score_rows.push_back(std::move(row_ranking[0]));
        index_rows.push_back(std::move(row_ranking[1]));
        masked_score_rows.push_back(std::move(row_ranking[2]));
      }
      ranking.emplace_back(
          spu::kernel::hal::concatenate(spu_ctx, score_rows, 0));
      ranking.emplace_back(
          spu::kernel::hal::concatenate(spu_ctx, index_rows, 0));
      ranking.emplace_back(
          spu::kernel::hal::concatenate(spu_ctx, masked_score_rows, 0));
    } else {
      SPDLOG_INFO(
          "CryptoMoE expert-wise Top-t mode=parallel: batching SPU "
          "QuickSelect partition rounds with per-round longest-interval "
          "padding");
      auto priority_scores =
          spu::kernel::hal::concatenate(spu_ctx, priority_scores_vec, 0);
      auto masked_scores =
          spu::kernel::hal::concatenate(spu_ctx, masked_scores_vec, 0);
      ranking = customized::batched_quick_select_topk_2d(
          spu_ctx, priority_scores, {masked_scores}, select_expert_nums);
    }
    SPU_ENFORCE(ranking.size() == 3);

    auto div_factor = spu::kernel::hlo::Constant(spu_ctx, active_expert_nums,
                                                 ranking[1].shape());

    // QuickSelect compares only R = M + W.  S = M * W is shuffled and swapped
    // as a payload using exactly the same permutation/partition decisions, so
    // the first t payload entries are already the weights consumed by Combine.
    auto selected_token_scores = std::move(ranking[2]);
    auto selected_token_indices =
        spu::kernel::hlo::Div(spu_ctx, ranking[1], div_factor);
    COMM_TIMER_END_PRINT(lctx, CRYPTOMOE_EXPERT_WISE_TOP_T);
    return std::make_pair(std::move(selected_token_scores),
                          std::move(selected_token_indices));
  }();
  auto all_selected_token_scores = std::move(selected_tokens.first);
  auto all_selected_token_indices = std::move(selected_tokens.second);

  for (size_t i = 0; i < config.n_; ++i) {
    auto selected_token_scores = spu::kernel::hal::reshape(
        spu_ctx,
        spu::kernel::hal::slice(
            spu_ctx, all_selected_token_scores, {static_cast<int64_t>(i), 0},
            {static_cast<int64_t>(i + 1), select_expert_nums}),
        {select_expert_nums});
    auto selected_token_indices = spu::kernel::hal::reshape(
        spu_ctx,
        spu::kernel::hal::slice(
            spu_ctx, all_selected_token_indices, {static_cast<int64_t>(i), 0},
            {static_cast<int64_t>(i + 1), select_expert_nums}),
        {select_expert_nums});

    // auto public_indices = spu::kernel::hal::dump_public_as<uint64_t>(
    //     spu_ctx, spu::kernel::hal::reveal(spu_ctx, selected_token_indices));
    // SPDLOG_INFO("{}", std::vector<uint64_t>(public_indices.begin(),
    //                                         public_indices.end()));

    auto batch_onehot = [&] {
      COMM_TIMER_START(lctx, CRYPTOMOE_ONE_HOT_MATRIX_GENERATION);
      auto result = BatchOneHot(spu_ctx, selected_token_indices, token_nums);
      COMM_TIMER_END_PRINT(lctx, CRYPTOMOE_ONE_HOT_MATRIX_GENERATION);
      return result;
    }();

    //  auto public_onehot = spu::kernel::hal::dump_public_as<uint64_t>(
    //     spu_ctx, spu::kernel::hal::reveal(spu_ctx, batch_onehot));
    // SPDLOG_INFO("{}", std::vector<uint64_t>(public_onehot.begin(),
    //                                         public_onehot.end()));

    SPDLOG_INFO("batch_onehot shape {}", batch_onehot.shape());
    SPDLOG_INFO("x shape {}", x.shape());
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
        spu_ctx, selected_token_scores, {1, select_expert_nums}));
    selected_token_indices_vec.push_back(spu::kernel::hal::reshape(
        spu_ctx, batch_onehot, {1, select_expert_nums, token_nums}));
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
    SPDLOG_INFO("--- DispatchImpl DONE --- time: {} ms", second * 1000);
  }
  return {selected_embeddings_for_experts, selected_token_scores,
          selected_token_indices};
}

DispatchResult DispatchKernel(spu::SPUContext* spu_ctx, const spu::Value& x,
                              const spu::Value& expert_scores,
                              const spu::Value& expert_indices,
                              const DispatchConfig& config) {
  return DispatchKernelTimed(spu_ctx, x, expert_scores, expert_indices, config,
                             /*emit_timing=*/true);
}

namespace {

void DispatchImpl(spu::SPUContext* spu_ctx, spu::device::ColocatedIo* cio_ptr,
                  const DispatchConfig& config) {
  auto start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("--- DispatchImpl START ---");

  auto result = DispatchKernelTimed(
      spu_ctx, cio_ptr->deviceGetVar(config.kTokenEmbeddings),
      cio_ptr->deviceGetVar(config.kExpertScores),
      cio_ptr->deviceGetVar(config.kExpertIndices), config,
      /*emit_timing=*/false);
  cio_ptr->deviceSetVar(config.kSelectedEmbeddingsForExperts,
                        result.selected_embeddings);
  cio_ptr->deviceSetVar(config.kTopTScoresForExperts, result.selected_scores);
  cio_ptr->deviceSetVar(config.kOneHotTokenIndicesForExperts,
                        result.token_one_hot);
  auto end = std::chrono::high_resolution_clock::now();
  auto second =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();
  SPDLOG_INFO("--- DispatchImpl DONE --- time: {} ms", second * 1000);
}

}  // namespace

// =====================================================================================================================

void DispatchServer::FeedInputs(
    const std::vector<double>& token_embeddings /* A.K.A. x */,
    const std::vector<uint64_t>& flatten_expert_indices /* A.K.A. K */,
    const std::vector<double>& flatten_expert_scores /* A.K.A. W */) {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  SPDLOG_INFO("Server FeedInputs");
  SPDLOG_INFO("Server FeedInputs");

  auto& spu_ctx_ = env_ptr_->spu_ctx_;
  auto& cio_ = env_ptr_->cio_;

  auto token_nums = config_.m_;
  auto active_expert_nums = config_.k_;
  auto embedding_size = token_embeddings.size() / token_nums;

  // Make copies to get mutable adapters for hostSetVar
  auto token_embeddings_copy = token_embeddings;
  auto flatten_expert_indices_copy = flatten_expert_indices;
  auto flatten_expert_scores_copy = flatten_expert_scores;

  auto xt_token_embeddings =
      xt::adapt(token_embeddings_copy, {token_nums, embedding_size});
  auto xt_expert_indices =
      xt::adapt(flatten_expert_indices_copy, {token_nums, active_expert_nums});
  auto xt_expert_scores =
      xt::adapt(flatten_expert_scores_copy, {token_nums, active_expert_nums});

  cio_->hostSetVar(fmt::format("server_{}", config_.kTokenEmbeddings),
                   xt_token_embeddings);
  cio_->hostSetVar(fmt::format("server_{}", config_.kExpertIndices),
                   xt_expert_indices);
  cio_->hostSetVar(fmt::format("server_{}", config_.kExpertScores),
                   xt_expert_scores);
  cio_->sync();

  auto lambda = [&](const std::string& var_name) {
    auto server_var = cio_->deviceGetVar(fmt::format("server_{}", var_name));
    auto user_var = cio_->deviceGetVar(fmt::format("user_{}", var_name));
    return spu::kernel::hlo::Add(spu_ctx_.get(), server_var, user_var);
  };

  cio_->deviceSetVar(config_.kTokenEmbeddings,
                     lambda(config_.kTokenEmbeddings));
  cio_->deviceSetVar(config_.kExpertIndices, lambda(config_.kExpertIndices));
  cio_->deviceSetVar(config_.kExpertScores, lambda(config_.kExpertScores));
}

void DispatchServer::Compute() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  DispatchImpl(env_ptr_->spu_ctx_.get(), env_ptr_->cio_.get(), config_);
}

std::vector<double> DispatchServer::ExtractOutput() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  auto xt_selected_embeddings = spu::kernel::hal::dump_public_as<double>(
      env_ptr_->spu_ctx_.get(),
      spu::kernel::hal::reveal(
          env_ptr_->spu_ctx_.get(),
          env_ptr_->cio_->deviceGetVar(config_.kSelectedEmbeddingsForExperts)));

  std::vector<double> selected_embeddings(xt_selected_embeddings.begin(),
                                          xt_selected_embeddings.end());
  return selected_embeddings;

  // std::vector<uint64_t> selected_embeddings;
  // for (size_t i = 0; i < config_.n_; ++i) {
  //   auto embedding = spu::kernel::hal::dump_public_as<uint64_t>(
  //       env_ptr_->spu_ctx_.get(),
  //       spu::kernel::hal::reveal(env_ptr_->spu_ctx_.get(),
  //                                env_ptr_->cio_->deviceGetVar(fmt::format(
  //                                    "selected_embeddings_for_expert_{}",
  //                                    i))));

  //   std::copy(embedding.begin(), embedding.end(),
  //             std::back_inserter(selected_embeddings));
  // }
  // return selected_embeddings;
}

// =====================================================================================================================

void DispatchUser::FeedInputs(
    const std::vector<double>& token_embeddings /* A.K.A. x */,
    const std::vector<uint64_t>& flatten_expert_indices /* A.K.A. K */,
    const std::vector<double>& flatten_expert_scores /* A.K.A. W */) {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  SPDLOG_INFO("User FeedInputs");

  auto& spu_ctx_ = env_ptr_->spu_ctx_;
  auto& cio_ = env_ptr_->cio_;

  auto token_nums = config_.m_;
  auto active_expert_nums = config_.k_;
  auto embedding_size = token_embeddings.size() / token_nums;

  // Make copies to get mutable adapters for hostSetVar
  auto token_embeddings_copy = token_embeddings;
  auto flatten_expert_indices_copy = flatten_expert_indices;
  auto flatten_expert_scores_copy = flatten_expert_scores;

  auto xt_token_embeddings =
      xt::adapt(token_embeddings_copy, {token_nums, embedding_size});
  auto xt_expert_indices =
      xt::adapt(flatten_expert_indices_copy, {token_nums, active_expert_nums});
  auto xt_expert_scores =
      xt::adapt(flatten_expert_scores_copy, {token_nums, active_expert_nums});

  cio_->hostSetVar(fmt::format("user_{}", config_.kTokenEmbeddings),
                   xt_token_embeddings);
  cio_->hostSetVar(fmt::format("user_{}", config_.kExpertIndices),
                   xt_expert_indices);
  cio_->hostSetVar(fmt::format("user_{}", config_.kExpertScores),
                   xt_expert_scores);
  cio_->sync();

  auto lambda = [&](const std::string& var_name) {
    auto server_var = cio_->deviceGetVar(fmt::format("server_{}", var_name));
    auto user_var = cio_->deviceGetVar(fmt::format("user_{}", var_name));
    return spu::kernel::hlo::Add(spu_ctx_.get(), server_var, user_var);
  };

  cio_->deviceSetVar(config_.kTokenEmbeddings,
                     lambda(config_.kTokenEmbeddings));
  cio_->deviceSetVar(config_.kExpertIndices, lambda(config_.kExpertIndices));
  cio_->deviceSetVar(config_.kExpertScores, lambda(config_.kExpertScores));
}

void DispatchUser::Compute() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  DispatchImpl(env_ptr_->spu_ctx_.get(), env_ptr_->cio_.get(), config_);
}

std::vector<double> DispatchUser::ExtractOutput() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  auto xt_selected_embeddings = spu::kernel::hal::dump_public_as<double>(
      env_ptr_->spu_ctx_.get(),
      spu::kernel::hal::reveal(
          env_ptr_->spu_ctx_.get(),
          env_ptr_->cio_->deviceGetVar(config_.kSelectedEmbeddingsForExperts)));

  std::vector<double> selected_embeddings(xt_selected_embeddings.begin(),
                                          xt_selected_embeddings.end());
  return selected_embeddings;

  // for (size_t i = 0; i < config_.n_; ++i) {
  //   auto embedding = spu::kernel::hal::dump_public_as<uint64_t>(
  //       env_ptr_->spu_ctx_.get(),
  //       spu::kernel::hal::reveal(env_ptr_->spu_ctx_.get(),
  //                                env_ptr_->cio_->deviceGetVar(fmt::format(
  //                                    "selected_embeddings_for_expert_{}",
  //                                    i))));

  //   std::copy(embedding.begin(), embedding.end(),
  //             std::back_inserter(selected_embeddings));
  // }
  // return selected_embeddings;
}

}  // namespace capricorn
