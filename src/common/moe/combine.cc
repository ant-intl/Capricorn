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

#include "src/common/moe/combine.h"

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

static spu::Value CombineKernelTimed(spu::SPUContext* spu_ctx,
                                     const spu::Value& y, const spu::Value& S,
                                     const spu::Value& K,
                                     const CombineConfig& config,
                                     bool emit_timing) {
  auto start = std::chrono::high_resolution_clock::now();
  if (emit_timing) {
    SPDLOG_INFO("--- Combine START ---");
  }

  YACL_ENFORCE(y.shape() == spu::Shape({static_cast<int64_t>(config.n_),
                                        static_cast<int64_t>(config.t_),
                                        static_cast<int64_t>(config.d_)}));
  YACL_ENFORCE(S.shape() == spu::Shape({static_cast<int64_t>(config.n_),
                                        static_cast<int64_t>(config.t_)}));
  YACL_ENFORCE(K.shape() == spu::Shape({static_cast<int64_t>(config.n_),
                                        static_cast<int64_t>(config.t_),
                                        static_cast<int64_t>(config.m_)}));

  int64_t n = config.n_;
  int64_t m = config.m_;
  int64_t t = config.t_;
  int64_t d = config.d_;

  auto combined_embeddings =
      spu::kernel::hal::zeros(spu_ctx, y.dtype(), {m, d});

  for (int64_t i = 0; i < n; ++i) {
    auto onehot_i =
        spu::kernel::hal::slice(spu_ctx, K, {i, 0, 0}, {i + 1, t, m}, {});

    // auto _onehot_i = spu::kernel::hal::dump_public_as<uint64_t>(
    //     spu_ctx, spu::kernel::hal::reveal(spu_ctx, onehot_i));
    // SPDLOG_INFO("shape : {}, _onehot_i : {}", onehot_i.shape(),
    //             std::vector<uint64_t>(_onehot_i.begin(), _onehot_i.end()));

    auto onehot_i_T = spu::kernel::hal::transpose(
        spu_ctx, spu::kernel::hal::reshape(spu_ctx, onehot_i, {t, m}), {1, 0});

    // auto _onehot_i_T = spu::kernel::hal::dump_public_as<uint64_t>(
    //     spu_ctx, spu::kernel::hal::reveal(spu_ctx, onehot_i_T));
    // SPDLOG_INFO("shape : {}, _onehot_i_T : {}", onehot_i_T.shape(),
    //             std::vector<uint64_t>(_onehot_i_T.begin(),
    //             _onehot_i_T.end()));

    auto S_i = spu::kernel::hal::slice(spu_ctx, S, {i, 0}, {i + 1, t}, {});
    auto y_Ei =
        spu::kernel::hal::slice(spu_ctx, y, {i, 0, 0}, {i + 1, t, d}, {});
    auto y_E = spu::kernel::hal::reshape(spu_ctx, y_Ei, {t, d});
    // CryptoMoE Algorithm 2: weight each column of the transposed one-hot
    // matrix before multiplying by expert outputs. This scores m*t entries
    // instead of t*d expert-output entries.
    auto scored_onehot = spu::kernel::hlo::Mul(
        spu_ctx, onehot_i_T,
        spu::kernel::hal::broadcast_to(spu_ctx, S_i, {m, t}));
    auto y_i = spu::kernel::hlo::Dot(spu_ctx, scored_onehot, y_E);

    // auto _y_i = spu::kernel::hal::dump_public_as<uint64_t>(
    //     spu_ctx, spu::kernel::hal::reveal(spu_ctx, y_i));
    // SPDLOG_INFO("shape : {}, y_i : {}", y_i.shape(),
    //             std::vector<uint64_t>(_y_i.begin(), _y_i.end()));

    combined_embeddings =
        spu::kernel::hlo::Add(spu_ctx, combined_embeddings, y_i);
  }

  // auto _combined_embeddings = spu::kernel::hal::dump_public_as<uint64_t>(
  //     spu_ctx, spu::kernel::hal::reveal(spu_ctx, combined_embeddings));
  // SPDLOG_INFO("shape : {}, combined_embeddings : {}",
  //             combined_embeddings.shape(),
  //             std::vector<uint64_t>(_combined_embeddings.begin(),
  //                                   _combined_embeddings.end()));

  auto end = std::chrono::high_resolution_clock::now();
  auto second =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();
  if (emit_timing) {
    SPDLOG_INFO("--- Combine DONE --- time: {} ms", second * 1000);
  }
  return combined_embeddings;
}

spu::Value CombineKernel(spu::SPUContext* spu_ctx, const spu::Value& y,
                         const spu::Value& S, const spu::Value& K,
                         const CombineConfig& config) {
  return CombineKernelTimed(spu_ctx, y, S, K, config, /*emit_timing=*/true);
}

namespace {

void CombineImpl(spu::SPUContext* spu_ctx, spu::device::ColocatedIo* cio_ptr,
                 const CombineConfig& config) {
  auto start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("--- Combine START ---");

  auto output = CombineKernelTimed(
      spu_ctx, cio_ptr->deviceGetVar(config.kExpertsOutEmbeddings),
      cio_ptr->deviceGetVar(config.kTopTScoresForExperts),
      cio_ptr->deviceGetVar(config.kOneHotTokenIndicesForExperts), config,
      /*emit_timing=*/false);
  cio_ptr->deviceSetVar(config.kCombinedEmbeddings, output);
  auto end = std::chrono::high_resolution_clock::now();
  auto second =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();
  SPDLOG_INFO("--- Combine DONE --- time: {} ms", second * 1000);
}

}  // namespace

// =====================================================================================================================

void CombineServer::FeedInputs(
    const std::vector<double>& experts_out_embeddings,
    const std::vector<double>& top_t_scores_for_experts,
    const std::vector<uint64_t>& one_hot_token_indices_for_experts) {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  SPDLOG_INFO("Server FeedInputs");

  auto& spu_ctx_ = env_ptr_->spu_ctx_;
  auto& cio_ = env_ptr_->cio_;

  auto expert_num = config_.n_;
  auto token_nums = config_.m_;
  auto embedding_dimension = config_.d_;
  auto active_token_per_experts = config_.t_;

  auto experts_out_embeddings_copy = experts_out_embeddings;
  auto top_t_scores_for_experts_copy = top_t_scores_for_experts;
  auto one_hot_token_indices_for_experts_copy =
      one_hot_token_indices_for_experts;

  auto xt_experts_out_embeddings =
      xt::adapt(experts_out_embeddings_copy,
                {expert_num, active_token_per_experts, embedding_dimension});
  auto xt_top_t_scores_for_experts = xt::adapt(
      top_t_scores_for_experts_copy, {expert_num, active_token_per_experts});
  auto xt_one_hot_token_indices_for_experts =
      xt::adapt(one_hot_token_indices_for_experts_copy,
                {expert_num, active_token_per_experts, token_nums});

  cio_->hostSetVar(fmt::format("server_{}", config_.kExpertsOutEmbeddings),
                   xt_experts_out_embeddings);
  cio_->hostSetVar(fmt::format("server_{}", config_.kTopTScoresForExperts),
                   xt_top_t_scores_for_experts);
  cio_->hostSetVar(
      fmt::format("server_{}", config_.kOneHotTokenIndicesForExperts),
      xt_one_hot_token_indices_for_experts);
  cio_->sync();

  auto lambda = [&](const std::string& var_name) {
    auto server_var = cio_->deviceGetVar(fmt::format("server_{}", var_name));
    auto user_var = cio_->deviceGetVar(fmt::format("user_{}", var_name));
    return spu::kernel::hlo::Add(spu_ctx_.get(), server_var, user_var);
  };

  cio_->deviceSetVar(config_.kExpertsOutEmbeddings,
                     lambda(config_.kExpertsOutEmbeddings));
  cio_->deviceSetVar(config_.kTopTScoresForExperts,
                     lambda(config_.kTopTScoresForExperts));
  cio_->deviceSetVar(config_.kOneHotTokenIndicesForExperts,
                     lambda(config_.kOneHotTokenIndicesForExperts));
}

void CombineServer::Compute() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  CombineImpl(env_ptr_->spu_ctx_.get(), env_ptr_->cio_.get(), config_);
}

std::vector<double> CombineServer::ExtractOutput() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  auto xt_combined_embeddings = spu::kernel::hal::dump_public_as<double>(
      env_ptr_->spu_ctx_.get(),
      spu::kernel::hal::reveal(
          env_ptr_->spu_ctx_.get(),
          env_ptr_->cio_->deviceGetVar(config_.kCombinedEmbeddings)));

  std::vector<double> combined_embeddings(xt_combined_embeddings.begin(),
                                          xt_combined_embeddings.end());
  return combined_embeddings;
}

// =====================================================================================================================

void CombineUser::FeedInputs(
    const std::vector<double>& experts_out_embeddings,
    const std::vector<double>& top_t_scores_for_experts,
    const std::vector<uint64_t>& one_hot_token_indices_for_experts) {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  SPDLOG_INFO("Server FeedInputs");

  auto& spu_ctx_ = env_ptr_->spu_ctx_;
  auto& cio_ = env_ptr_->cio_;

  auto expert_num = config_.n_;
  auto token_nums = config_.m_;
  auto embedding_dimension = config_.d_;
  auto active_token_per_experts = config_.t_;

  auto experts_out_embeddings_copy = experts_out_embeddings;
  auto top_t_scores_for_experts_copy = top_t_scores_for_experts;
  auto one_hot_token_indices_for_experts_copy =
      one_hot_token_indices_for_experts;

  auto xt_experts_out_embeddings =
      xt::adapt(experts_out_embeddings_copy,
                {expert_num, active_token_per_experts, embedding_dimension});
  auto xt_top_t_scores_for_experts = xt::adapt(
      top_t_scores_for_experts_copy, {expert_num, active_token_per_experts});
  auto xt_one_hot_token_indices_for_experts =
      xt::adapt(one_hot_token_indices_for_experts_copy,
                {expert_num, active_token_per_experts, token_nums});

  cio_->hostSetVar(fmt::format("user_{}", config_.kExpertsOutEmbeddings),
                   xt_experts_out_embeddings);
  cio_->hostSetVar(fmt::format("user_{}", config_.kTopTScoresForExperts),
                   xt_top_t_scores_for_experts);
  cio_->hostSetVar(
      fmt::format("user_{}", config_.kOneHotTokenIndicesForExperts),
      xt_one_hot_token_indices_for_experts);
  cio_->sync();

  auto lambda = [&](const std::string& var_name) {
    auto server_var = cio_->deviceGetVar(fmt::format("server_{}", var_name));
    auto user_var = cio_->deviceGetVar(fmt::format("user_{}", var_name));
    return spu::kernel::hlo::Add(spu_ctx_.get(), server_var, user_var);
  };

  cio_->deviceSetVar(config_.kExpertsOutEmbeddings,
                     lambda(config_.kExpertsOutEmbeddings));
  cio_->deviceSetVar(config_.kTopTScoresForExperts,
                     lambda(config_.kTopTScoresForExperts));
  cio_->deviceSetVar(config_.kOneHotTokenIndicesForExperts,
                     lambda(config_.kOneHotTokenIndicesForExperts));
}

void CombineUser::Compute() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  CombineImpl(env_ptr_->spu_ctx_.get(), env_ptr_->cio_.get(), config_);
}

std::vector<double> CombineUser::ExtractOutput() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  auto xt_combined_embeddings = spu::kernel::hal::dump_public_as<double>(
      env_ptr_->spu_ctx_.get(),
      spu::kernel::hal::reveal(
          env_ptr_->spu_ctx_.get(),
          env_ptr_->cio_->deviceGetVar(config_.kCombinedEmbeddings)));

  std::vector<double> combined_embeddings(xt_combined_embeddings.begin(),
                                          xt_combined_embeddings.end());
  return combined_embeddings;
}

}  // namespace capricorn
