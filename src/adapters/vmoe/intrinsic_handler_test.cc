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

#include "src/adapters/vmoe/intrinsic_handler.h"

#include "gtest/gtest.h"
#include "src/adapters/vmoe/expert_choice_moe_block.h"
#include "src/adapters/vmoe/sparse_moe_block.h"

namespace capricorn::vmoe {
namespace {

TEST(IntrinsicHandlerTest, RecognizesOnlyVersionedVmoeTarget) {
  EXPECT_TRUE(IsVmoeIntrinsicTarget(kVmoeS32SiluTarget));
  EXPECT_TRUE(IsVmoeIntrinsicTarget(kVmoeS32ExpertChoiceSiluTarget));
  EXPECT_TRUE(IsVmoeIntrinsicTarget(kVmoeS32GeluBiasTarget));
  EXPECT_TRUE(IsVmoeIntrinsicTarget(kVmoeS32ExpertChoiceGeluBiasTarget));
  EXPECT_TRUE(IsVmoeIntrinsicTarget(kVmoeS32SiluBiasTarget));
  EXPECT_TRUE(IsVmoeIntrinsicTarget(kVmoeS32ExpertChoiceSecureSiluBiasTarget));
  EXPECT_FALSE(IsVmoeIntrinsicTarget("mhlo.topk"));
  EXPECT_FALSE(IsVmoeIntrinsicTarget("spu.capricorn.vmoe_s32_e8_k2_silu"));
}

TEST(IntrinsicHandlerTest, MatchesPatchedSpuExtensionPoint) {
  auto handler = MakeVmoeIntrinsicHandler(nullptr);
  EXPECT_TRUE(static_cast<bool>(handler));
}

TEST(IntrinsicHandlerTest, RecognizesAndParsesOnlySafeTraceTargets) {
  EXPECT_TRUE(
      IsVmoeTraceTarget("spu.capricorn.vmoe_trace.block_5.self_attention.v1"));
  EXPECT_EQ(
      ParseVmoeTraceStage("spu.capricorn.vmoe_trace.block_5.self_attention.v1"),
      "block_5.self_attention");
  EXPECT_EQ(ParseVmoeTraceStage("spu.capricorn.vmoe_trace.end.v1"), "end");
  EXPECT_FALSE(IsVmoeTraceTarget("spu.capricorn.vmoe_trace..v1"));
  EXPECT_FALSE(
      IsVmoeTraceTarget("spu.capricorn.vmoe_trace.block 5/attention.v1"));
  EXPECT_FALSE(
      IsVmoeTraceTarget("spu.capricorn.vmoe_trace.block_5.self_attention.v2"));
}

}  // namespace
}  // namespace capricorn::vmoe
