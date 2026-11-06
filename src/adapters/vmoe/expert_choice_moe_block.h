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

#include "libspu/core/context.h"
#include "libspu/core/value.h"
#include "src/adapters/vmoe/config.h"

namespace capricorn::silu {
class SecureSiluSession;
}

namespace capricorn::vmoe {

inline constexpr char kVmoeS32ExpertChoiceSiluTarget[] =
    "spu.capricorn.vmoe_s32_e8_t16_silu.expert_choice.v1";
inline constexpr char kVmoeS32ExpertChoiceGeluBiasTarget[] =
    "spu.capricorn.vmoe_s32_e8_t28_gelu_bias.expert_choice.v2";
inline constexpr char kVmoeS32ExpertChoiceSecureSiluBiasTarget[] =
    "spu.capricorn.vmoe_s32_e8_t28_secure_silu_bias.expert_choice.v3";

spu::Value RunExpertChoiceMoeBlock(
    spu::SPUContext* spu_ctx, const spu::Value& token_embeddings,
    const spu::Value& router_weights, const spu::Value& experts_weights_one,
    const spu::Value& experts_weights_two, const ExpertChoiceMoeConfig& config,
    silu::SecureSiluSession* silu_session = nullptr,
    const spu::Value* token_mask = nullptr);

spu::Value RunExpertChoiceMoeBlockWithBias(
    spu::SPUContext* spu_ctx, const spu::Value& token_embeddings,
    const spu::Value& router_weights, const spu::Value& experts_weights_one,
    const spu::Value& experts_bias_one, const spu::Value& experts_weights_two,
    const spu::Value& experts_bias_two, const ExpertChoiceMoeConfig& config);

spu::Value RunExpertChoiceMoeBlockWithBiasSecureSilu(
    spu::SPUContext* spu_ctx, const spu::Value& token_embeddings,
    const spu::Value& router_weights, const spu::Value& experts_weights_one,
    const spu::Value& experts_bias_one, const spu::Value& experts_weights_two,
    const spu::Value& experts_bias_two, const ExpertChoiceMoeConfig& config,
    silu::SecureSiluSession* silu_session = nullptr);

}  // namespace capricorn::vmoe
