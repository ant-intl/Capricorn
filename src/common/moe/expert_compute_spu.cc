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

#include "src/common/moe/expert_compute_spu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "libspu/core/config.h"
#include "libspu/core/encoding.h"
#include "libspu/core/half.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "src/common/he/batch_matmul.h"
#include "src/common/nn/spu_activation.h"
// #include "libspu/core/type.h"
#include "libspu/device/io.h"
#include "libspu/kernel/hal/constants.h"  // bad reference
#include "libspu/kernel/hal/intrinsic/nn/activation.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/ring.h"
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
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "spdlog/spdlog.h"
#include "src/common/runtime/macro.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xcsv.hpp"
#include "xtensor/views/xview.hpp"

namespace capricorn {

namespace {

// Multiply secret-shared activations by one party's private expert weights.
// A supplied session keeps benchmark setup out of the online stage; otherwise
// initialize an independent one-shot session as in the ordinary kernel path.
spu::NdArrayRef BatchMatMulAV(
    std::shared_ptr<spu::mpc::cheetah::BatchMatMul> session,
    const std::shared_ptr<yacl::link::Context>& lctx, spu::FieldType field,
    const spu::NdArrayRef& x, const spu::NdArrayRef& y, bool y_owner) {
  const spu::mpc::cheetah::Shape4D dim4 = {x.shape()[0], x.shape()[1],
                                           x.shape()[2], y.shape()[2]};
  std::shared_ptr<yacl::link::Context> one_time_link;
  if (session == nullptr) {
    one_time_link = lctx->Spawn();
    session = std::make_shared<spu::mpc::cheetah::BatchMatMul>(one_time_link,
                                                               y_owner, false);
    session->LazyInitKeys(field);
  } else {
    SPU_ENFORCE(session->Rank() == static_cast<int>(lctx->Rank()));
    SPU_ENFORCE(
        session->IsPreparedForShape(field, dim4),
        "persistent BatchMatMul session was not prepared for this shape");
  }

  const auto& local_input = y_owner ? y : x;
  auto output =
      session->BatchDotOLE(local_input, one_time_link.get(), dim4, !y_owner);
  output = output.reshape({dim4[0], dim4[1], dim4[3]});
  if (y_owner) {
    // OLE covers the other party's activation share. Add the product of
    // the weight owner's local activation share and its private weights.
    for (int64_t batch = 0; batch < dim4[0]; ++batch) {
      auto lhs = x.slice({batch, 0, 0}, {batch + 1, dim4[1], dim4[2]}, {})
                     .reshape({dim4[1], dim4[2]});
      auto rhs = y.slice({batch, 0, 0}, {batch + 1, dim4[2], dim4[3]}, {})
                     .reshape({dim4[2], dim4[3]});
      auto local = spu::mpc::ring_mmul(lhs, rhs);
      auto output_slice =
          output.slice({batch, 0, 0}, {batch + 1, dim4[1], dim4[3]}, {})
              .reshape({dim4[1], dim4[3]});
      spu::mpc::ring_add_(output_slice, local);
    }
  }
  return output.as(x.eltype());
}

spu::Value BatchExpertComputeKernelImpl(
    spu::SPUContext* spu_ctx, const spu::Value& x, const spu::Value& EW1,
    const spu::Value* EB1, const spu::Value& EW2, const spu::Value* EB2,
    const BatchExpertComputeConfig& config, ExpertActivation activation,
    bool emit_timing) {
  auto start = std::chrono::high_resolution_clock::now();
  if (emit_timing) {
    SPDLOG_INFO("--- ExpertCompute START ---");
  }

  YACL_ENFORCE(x.shape() == spu::Shape({static_cast<int64_t>(config.n_),
                                        static_cast<int64_t>(config.t_),
                                        static_cast<int64_t>(config.d_)}));
  YACL_ENFORCE(EW1.shape() == spu::Shape({static_cast<int64_t>(config.n_),
                                          static_cast<int64_t>(config.d_),
                                          static_cast<int64_t>(config.ed_)}));
  YACL_ENFORCE(EW2.shape() == spu::Shape({static_cast<int64_t>(config.n_),
                                          static_cast<int64_t>(config.ed_),
                                          static_cast<int64_t>(config.d_)}));
  if (EB1 != nullptr) {
    YACL_ENFORCE(EB1->shape() ==
                 spu::Shape({static_cast<int64_t>(config.n_),
                             static_cast<int64_t>(config.ed_)}));
  }
  if (EB2 != nullptr) {
    YACL_ENFORCE(EB2->shape() == spu::Shape({static_cast<int64_t>(config.n_),
                                             static_cast<int64_t>(config.d_)}));
  }

  YACL_ENFORCE(EW1.isPrivate() && EW2.isPrivate());
  YACL_ENFORCE(EB1 == nullptr || EB1->isPrivate());
  YACL_ENFORCE(EB2 == nullptr || EB2->isPrivate());
  YACL_ENFORCE(x.isSecret());

  auto nd_x = x.data();
  auto nd_EW1 = EW1.data();
  auto nd_EW2 = EW2.data();

  const auto* w1_ptype = nd_EW1.eltype().as<spu::mpc::Priv2kTy>();
  SPU_ENFORCE(w1_ptype != nullptr, "weight one should be a private type");
  const auto* w2_ptype = nd_EW2.eltype().as<spu::mpc::Priv2kTy>();
  SPU_ENFORCE(w2_ptype != nullptr, "weight two should be a private type");
  YACL_ENFORCE(w1_ptype->owner() == w2_ptype->owner(),
               "weight one and two's owner should be the same");
  if (EB1 != nullptr) {
    const auto* b1_ptype = EB1->data().eltype().as<spu::mpc::Priv2kTy>();
    SPU_ENFORCE(b1_ptype != nullptr && b1_ptype->owner() == w1_ptype->owner(),
                "expert bias one must have the same private owner");
  }
  if (EB2 != nullptr) {
    const auto* b2_ptype = EB2->data().eltype().as<spu::mpc::Priv2kTy>();
    SPU_ENFORCE(b2_ptype != nullptr && b2_ptype->owner() == w1_ptype->owner(),
                "expert bias two must have the same private owner");
  }

  auto lctx = spu_ctx->lctx();
  bool weight_owner = size_t(w1_ptype->owner()) == size_t(lctx->Rank());
  auto field = spu_ctx->getField();

  auto trunc_wrapper = [&](const spu::Value& ins) {
    return spu::kernel::hal::_trunc(spu_ctx, ins, spu_ctx->getFxpBits())
        .setDtype(x.dtype());
  };

  COMM_TIMER_START(lctx, W1);
  auto nd_hidden_result =
      BatchMatMulAV(config.w1_session, lctx, field, nd_x, nd_EW1, weight_owner);
  auto hidden_result = trunc_wrapper(spu::Value(nd_hidden_result, x.dtype()));
  COMM_TIMER_END_PRINT(lctx, W1);

  if (EB1 != nullptr) {
    COMM_TIMER_START(lctx, ExpertBias1);
    auto bias = spu::kernel::hal::reshape(
        spu_ctx, *EB1,
        {static_cast<int64_t>(config.n_), 1, static_cast<int64_t>(config.ed_)});
    bias = spu::kernel::hal::broadcast_to(spu_ctx, bias, hidden_result.shape());
    hidden_result = spu::kernel::hal::add(spu_ctx, hidden_result, bias);
    COMM_TIMER_END_PRINT(lctx, ExpertBias1);
  }

  SPDLOG_INFO("FIGURE4_SILU_SHAPE shape={} numel={}",
              hidden_result.shape(), hidden_result.numel());

  // Keep the model's chosen activation. Only SiLU has a canonical SiLU
  // benchmark stage; a GELU evaluation must not be labelled as SiLU.
  spu::Value activation_result;
  if (activation == ExpertActivation::kGelu) {
    COMM_TIMER_START(lctx, GELU);
    activation_result =
        spu::kernel::hal::intrinsic::nn::f_seg3_gelu(spu_ctx, hidden_result);
    COMM_TIMER_END_PRINT(lctx, GELU);
  } else {
    COMM_TIMER_START(lctx, SiLU);
    activation_result = customized::SpuPiecewiseSiLU(spu_ctx, hidden_result);
    auto nd_result = activation_result.data();
    SPDLOG_INFO("BatchExpertComputeImpl: SILU activation done");
    COMM_TIMER_END_PRINT(lctx, SiLU);
    static_cast<void>(nd_result);
  }
  auto nd_activation_result = activation_result.data();

  COMM_TIMER_START(lctx, W2);
  auto nd_outputs = BatchMatMulAV(config.w2_session, lctx, field,
                                  nd_activation_result, nd_EW2, weight_owner);
  auto outputs = trunc_wrapper(spu::Value(nd_outputs, x.dtype()));
  COMM_TIMER_END_PRINT(lctx, W2);

  if (EB2 != nullptr) {
    COMM_TIMER_START(lctx, ExpertBias2);
    auto bias = spu::kernel::hal::reshape(
        spu_ctx, *EB2,
        {static_cast<int64_t>(config.n_), 1, static_cast<int64_t>(config.d_)});
    bias = spu::kernel::hal::broadcast_to(spu_ctx, bias, outputs.shape());
    outputs = spu::kernel::hal::add(spu_ctx, outputs, bias);
    COMM_TIMER_END_PRINT(lctx, ExpertBias2);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto second =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();
  if (emit_timing) {
    SPDLOG_INFO("--- ExpertCompute DONE --- time: {} ms", second * 1000);
  }
  return outputs;
}

}  // namespace

static spu::Value BatchExpertComputeKernelTimed(
    spu::SPUContext* spu_ctx, const spu::Value& x, const spu::Value& EW1,
    const spu::Value& EW2, const BatchExpertComputeConfig& config,
    bool emit_timing) {
  return BatchExpertComputeKernelImpl(spu_ctx, x, EW1, nullptr, EW2, nullptr,
                                      config, ExpertActivation::kSilu,
                                      emit_timing);
}

spu::Value BatchExpertComputeKernel(spu::SPUContext* spu_ctx,
                                    const spu::Value& x, const spu::Value& EW1,
                                    const spu::Value& EW2,
                                    const BatchExpertComputeConfig& config) {
  return BatchExpertComputeKernelTimed(spu_ctx, x, EW1, EW2, config,
                                       /*emit_timing=*/true);
}

spu::Value BatchExpertComputeKernelWithBias(
    spu::SPUContext* spu_ctx, const spu::Value& x, const spu::Value& EW1,
    const spu::Value& EB1, const spu::Value& EW2, const spu::Value& EB2,
    const BatchExpertComputeConfig& config, ExpertActivation activation) {
  return BatchExpertComputeKernelImpl(spu_ctx, x, EW1, &EB1, EW2, &EB2, config,
                                      activation, /*emit_timing=*/true);
}

namespace {

void BatchExpertComputeImpl(spu::SPUContext* spu_ctx,
                            spu::device::ColocatedIo* cio_ptr,
                            const BatchExpertComputeConfig& config) {
  auto start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("--- ExpertCompute START ---");
  auto outputs = BatchExpertComputeKernelTimed(
      spu_ctx, cio_ptr->deviceGetVar(config.kSelectedEmbeddingsForExperts),
      cio_ptr->deviceGetVar(config.kExpertsWeightsOne),
      cio_ptr->deviceGetVar(config.kExpertsWeightsTwo), config,
      /*emit_timing=*/false);
  cio_ptr->deviceSetVar(config.kExpertsOutEmbeddings, outputs);
  auto end = std::chrono::high_resolution_clock::now();
  auto second =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();
  SPDLOG_INFO("--- ExpertCompute DONE --- time: {} ms", second * 1000);
}

}  // namespace

// =====================================================================================================================

void BatchExpertComputeServer::FeedInputs(
    const std::vector<double>& selected_embeddings_for_experts,
    const std::vector<double>& experts_weights_one,
    const std::vector<double>& experts_weights_two) {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  SPDLOG_INFO("Server FeedInputs");

  auto& spu_ctx_ = env_ptr_->spu_ctx_;
  auto& cio_ = env_ptr_->cio_;

  auto expert_num = config_.n_;
  auto active_token_per_experts = config_.t_;
  auto embedding_size = config_.d_;
  auto expert_dimension = config_.ed_;

  // Make copies to get mutable adapters for hostSetVar
  YACL_ENFORCE(selected_embeddings_for_experts.size() ==
                   expert_num * active_token_per_experts * embedding_size,
               "Invalid input size");
  YACL_ENFORCE(experts_weights_one.size() ==
                   expert_num * embedding_size * expert_dimension,
               "Invalid weight one size");
  YACL_ENFORCE(experts_weights_two.size() ==
                   expert_num * expert_dimension * embedding_size,
               "Invalid weight two size");

  auto selected_embeddings_for_experts_copy = selected_embeddings_for_experts;
  auto experts_weights_one_copy = experts_weights_one;
  auto experts_weights_two_copy = experts_weights_two;

  auto xt_selected_embeddings_for_experts =
      xt::adapt(selected_embeddings_for_experts_copy,
                {expert_num, active_token_per_experts, embedding_size});

  auto xt_experts_weights_one = xt::adapt(
      experts_weights_one_copy, {expert_num, embedding_size, expert_dimension});

  auto xt_experts_weights_two = xt::adapt(
      experts_weights_two_copy, {expert_num, expert_dimension, embedding_size});

  cio_->hostSetVar(
      fmt::format("server_{}", config_.kSelectedEmbeddingsForExperts),
      xt_selected_embeddings_for_experts);

  SPDLOG_INFO("Server PRIVATE");
  cio_->hostSetVar(fmt::format("_{}", config_.kExpertsWeightsOne),
                   xt_experts_weights_one, spu::Visibility::VIS_SECRET);
  cio_->hostSetVar(fmt::format("_{}", config_.kExpertsWeightsTwo),
                   xt_experts_weights_two, spu::Visibility::VIS_SECRET);
  cio_->sync();

  SPDLOG_INFO("Server SYNC DONE");

  auto lambda = [&](const std::string& var_name) {
    auto server_var = cio_->deviceGetVar(fmt::format("server_{}", var_name));
    auto user_var = cio_->deviceGetVar(fmt::format("user_{}", var_name));
    return spu::kernel::hlo::Add(spu_ctx_.get(), server_var, user_var);
  };

  auto lambda_a2v = [&](const std::string& var_name) {
    auto _var = cio_->deviceGetVar(fmt::format("_{}", var_name));
    return spu::kernel::hlo::RevealTo(spu_ctx_.get(), _var,
                                      spu_ctx_->lctx()->Rank());
  };

  cio_->deviceSetVar(config_.kSelectedEmbeddingsForExperts,
                     lambda(config_.kSelectedEmbeddingsForExperts));

  cio_->deviceSetVar(config_.kExpertsWeightsOne,
                     lambda_a2v(config_.kExpertsWeightsOne));
  cio_->deviceSetVar(config_.kExpertsWeightsTwo,
                     lambda_a2v(config_.kExpertsWeightsTwo));
}

void BatchExpertComputeServer::Compute() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  BatchExpertComputeImpl(env_ptr_->spu_ctx_.get(), env_ptr_->cio_.get(),
                         config_);
}

std::vector<double> BatchExpertComputeServer::ExtractOutput() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  auto xt_experts_out_embeddings = spu::kernel::hal::dump_public_as<double>(
      env_ptr_->spu_ctx_.get(),
      spu::kernel::hal::reveal(
          env_ptr_->spu_ctx_.get(),
          env_ptr_->cio_->deviceGetVar(config_.kExpertsOutEmbeddings)));

  std::vector<double> experts_out_embeddings(xt_experts_out_embeddings.begin(),
                                             xt_experts_out_embeddings.end());
  return experts_out_embeddings;
}

// =====================================================================================================================

void BatchExpertComputeUser::FeedInputs(
    const std::vector<double>& selected_embeddings_for_experts) {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  SPDLOG_INFO("User FeedInputs");

  auto& spu_ctx_ = env_ptr_->spu_ctx_;
  auto& cio_ = env_ptr_->cio_;

  auto expert_num = config_.n_;
  auto active_token_per_experts = config_.t_;
  auto embedding_size = config_.d_;
  // auto expert_dimension = config_.ed_;

  // Make copies to get mutable adapters for hostSetVar
  YACL_ENFORCE(selected_embeddings_for_experts.size() ==
                   expert_num * active_token_per_experts * embedding_size,
               "Invalid input size");
  auto selected_embeddings_for_experts_copy = selected_embeddings_for_experts;

  auto xt_selected_embeddings_for_experts =
      xt::adapt(selected_embeddings_for_experts_copy,
                {expert_num, active_token_per_experts, embedding_size});

  cio_->hostSetVar(
      fmt::format("user_{}", config_.kSelectedEmbeddingsForExperts),
      xt_selected_embeddings_for_experts);
  cio_->sync();

  auto lambda = [&](const std::string& var_name) {
    auto server_var = cio_->deviceGetVar(fmt::format("server_{}", var_name));
    auto user_var = cio_->deviceGetVar(fmt::format("user_{}", var_name));
    return spu::kernel::hlo::Add(spu_ctx_.get(), server_var, user_var);
  };

  auto lambda_a2v = [&](const std::string& var_name) {
    auto _var = cio_->deviceGetVar(fmt::format("_{}", var_name));
    return spu::kernel::hlo::RevealTo(spu_ctx_.get(), _var,
                                      spu_ctx_->lctx()->NextRank());
  };

  cio_->deviceSetVar(config_.kSelectedEmbeddingsForExperts,
                     lambda(config_.kSelectedEmbeddingsForExperts));

  cio_->deviceSetVar(config_.kExpertsWeightsOne,
                     lambda_a2v(config_.kExpertsWeightsOne));
  cio_->deviceSetVar(config_.kExpertsWeightsTwo,
                     lambda_a2v(config_.kExpertsWeightsTwo));
}

void BatchExpertComputeUser::Compute() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  BatchExpertComputeImpl(env_ptr_->spu_ctx_.get(), env_ptr_->cio_.get(),
                         config_);
}

std::vector<double> BatchExpertComputeUser::ExtractOutput() {
  YACL_ENFORCE(IsEnvReady(), "Environment is not ready");
  auto xt_experts_out_embeddings = spu::kernel::hal::dump_public_as<double>(
      env_ptr_->spu_ctx_.get(),
      spu::kernel::hal::reveal(
          env_ptr_->spu_ctx_.get(),
          env_ptr_->cio_->deviceGetVar(config_.kExpertsOutEmbeddings)));

  std::vector<double> experts_out_embeddings(xt_experts_out_embeddings.begin(),
                                             xt_experts_out_embeddings.end());
  return experts_out_embeddings;
}

}  // namespace capricorn
