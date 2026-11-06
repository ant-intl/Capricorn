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

#include "src/baselines/cryptomoe/dispatch.h"
#include "src/capricorn/expert_compute.h"
#include "src/capricorn/routing.h"
#include "src/common/moe/combine.h"
#include "src/common/moe/expert_compute_spu.h"
#include "src/common/moe/token_topk.h"

#include "yacl/base/exception.h"

namespace capricorn::vmoe {

struct SparseMoeConfig {
  GateRoutingConfig gate;
  DispatchConfig dispatch;
  BatchExpertComputeConfig expert;
  CombineConfig combine;

  static SparseMoeConfig VmoeS32(
      ExpertTopTMode top_t_mode = ExpertTopTMode::SerialQuickSelect) {
    return {
        GateRoutingConfig(/*token_nums=*/50, /*expert_nums=*/8, /*top_k=*/2,
                          /*embedding_dimension=*/512),
        DispatchConfig(/*token_nums=*/50, /*expert_nums=*/8,
                       /*activated_experts_per_token=*/2,
                       /*dispatch_expert_per_token=*/16,
                       /*embedding_dimension=*/512, top_t_mode),
        BatchExpertComputeConfig(/*expert_nums=*/8,
                                 /*activated_experts_per_token=*/16,
                                 /*embedding_dimension=*/512,
                                 /*expert_dimension=*/2048),
        CombineConfig(/*token_nums=*/50, /*expert_nums=*/8,
                      /*dispatch_expert_per_token=*/16,
                      /*embedding_dimension=*/512),
    };
  }

  static SparseMoeConfig OfficialVmoeS32(
      ExpertTopTMode top_t_mode = ExpertTopTMode::SerialQuickSelect) {
    auto config = VmoeS32(top_t_mode);
    config.dispatch.t_ = 52;
    config.expert.t_ = 52;
    config.combine.t_ = 52;
    return config;
  }

  void Validate() const {
    YACL_ENFORCE(
        gate.m_ == dispatch.m_ && dispatch.m_ == combine.m_,
        "token dimensions must match: gate={}, dispatch={}, combine={}",
        gate.m_, dispatch.m_, combine.m_);
    YACL_ENFORCE(gate.n_ == dispatch.n_ && dispatch.n_ == expert.n_ &&
                     expert.n_ == combine.n_,
                 "expert dimensions must match: gate={}, dispatch={}, "
                 "expert={}, combine={}",
                 gate.n_, dispatch.n_, expert.n_, combine.n_);
    YACL_ENFORCE(gate.k_ == dispatch.k_,
                 "top-k dimensions must match: gate={}, dispatch={}", gate.k_,
                 dispatch.k_);
    YACL_ENFORCE(
        dispatch.t_ == expert.t_ && expert.t_ == combine.t_,
        "capacity dimensions must match: dispatch={}, expert={}, combine={}",
        dispatch.t_, expert.t_, combine.t_);
    YACL_ENFORCE(
        gate.d_ == expert.d_ && expert.d_ == combine.d_,
        "embedding dimensions must match: gate={}, expert={}, combine={}",
        gate.d_, expert.d_, combine.d_);
  }
};

struct ExpertChoiceMoeConfig {
  RoutingConfig routing;
  SiLUExpertComputeConfig expert;
  CombineConfig combine;

  static ExpertChoiceMoeConfig VmoeS32() {
    return {
        RoutingConfig(/*token_nums=*/50, /*expert_nums=*/8,
                      /*expert_capacity=*/16,
                      /*embedding_dimension=*/512),
        SiLUExpertComputeConfig(/*expert_nums=*/8,
                                /*token_capacity_per_expert=*/16,
                                /*embedding_dimension=*/512,
                                /*expert_dimension=*/2048),
        CombineConfig(/*token_nums=*/50, /*expert_nums=*/8,
                      /*dispatch_expert_per_token=*/16,
                      /*embedding_dimension=*/512),
    };
  }

  static ExpertChoiceMoeConfig OfficialVmoeS32() {
    auto config = VmoeS32();
    config.routing.t_ = 28;
    config.expert.t_ = 28;
    config.combine.t_ = 28;
    return config;
  }

  void Validate() const {
    YACL_ENFORCE(routing.m_ == combine.m_,
                 "token dimensions must match: routing={}, combine={}",
                 routing.m_, combine.m_);
    YACL_ENFORCE(
        routing.n_ == expert.n_ && expert.n_ == combine.n_,
        "expert dimensions must match: routing={}, expert={}, combine={}",
        routing.n_, expert.n_, combine.n_);
    YACL_ENFORCE(routing.t_ > 0 && routing.t_ <= routing.m_,
                 "expert capacity must be in [1, {}], got {}", routing.m_,
                 routing.t_);
    YACL_ENFORCE(
        routing.t_ == expert.t_ && expert.t_ == combine.t_,
        "capacity dimensions must match: routing={}, expert={}, combine={}",
        routing.t_, expert.t_, combine.t_);
    YACL_ENFORCE(
        routing.d_ == expert.d_ && expert.d_ == combine.d_,
        "embedding dimensions must match: routing={}, expert={}, combine={}",
        routing.d_, expert.d_, combine.d_);
  }
};

}  // namespace capricorn::vmoe
