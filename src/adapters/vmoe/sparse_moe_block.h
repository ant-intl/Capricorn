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

namespace capricorn::vmoe {

inline constexpr char kVmoeS32SiluTarget[] =
    "spu.capricorn.vmoe_s32_e8_k2_silu.v1";
inline constexpr char kVmoeS32GeluBiasTarget[] =
    "spu.capricorn.vmoe_s32_e8_k2_gelu_bias.eval.v2";
inline constexpr char kVmoeS32SiluBiasTarget[] =
    "spu.capricorn.vmoe_s32_e8_k2_silu_bias.eval.v3";

spu::Value RunSparseMoeBlock(spu::SPUContext* spu_ctx,
                             const spu::Value& token_embeddings,
                             const spu::Value& router_weights,
                             const spu::Value& experts_weights_one,
                             const spu::Value& experts_weights_two,
                             const SparseMoeConfig& config);

spu::Value RunSparseMoeBlockWithBias(
    spu::SPUContext* spu_ctx, const spu::Value& token_embeddings,
    const spu::Value& router_weights, const spu::Value& experts_weights_one,
    const spu::Value& experts_bias_one, const spu::Value& experts_weights_two,
    const spu::Value& experts_bias_two, const SparseMoeConfig& config,
    ExpertActivation activation = ExpertActivation::kGelu);

}  // namespace capricorn::vmoe
