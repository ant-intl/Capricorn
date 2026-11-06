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

#include "src/adapters/vmoe/expert_choice_moe_block.h"

#include <cmath>
#include <future>
#include <vector>

#include "gtest/gtest.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"
#include "src/capricorn/activation/secure_silu.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/test_utils.h"
#include "xtensor/containers/xadapt.hpp"

namespace capricorn::vmoe {
namespace {

TEST(ExpertChoiceMoeBlockTest, PublishesVersionedRealShapeContract) {
  EXPECT_STREQ(kVmoeS32ExpertChoiceSiluTarget,
               "spu.capricorn.vmoe_s32_e8_t16_silu.expert_choice.v1");
  const auto config = ExpertChoiceMoeConfig::VmoeS32();
  EXPECT_EQ(config.routing.m_, 50);
  EXPECT_EQ(config.routing.n_, 8);
  EXPECT_EQ(config.routing.t_, 16);
  EXPECT_EQ(config.routing.d_, 512);
  EXPECT_EQ(config.expert.ed_, 2048);
}

TEST(ExpertChoiceMoeBlockTest, MatchesNonzeroPlaintextSiluOracle) {
  ExpertChoiceMoeConfig config{
      RoutingConfig(/*token_nums=*/8, /*expert_nums=*/2,
                    /*expert_capacity=*/8, /*embedding_dimension=*/4),
      SiLUExpertComputeConfig(/*expert_nums=*/2,
                              /*token_capacity_per_expert=*/8,
                              /*embedding_dimension=*/4,
                              /*expert_dimension=*/4),
      CombineConfig(/*token_nums=*/8, /*expert_nums=*/2,
                    /*dispatch_expert_per_token=*/8,
                    /*embedding_dimension=*/4),
  };
  config.Validate();

  std::vector<double> tokens(32);
  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = static_cast<double>(i % 9) / 16.0 - 0.25;
  }
  const std::vector<double> zero_tokens(tokens.size(), 0.0);
  const std::vector<double> router(8, 0.0);  // Official [D, E] layout.
  std::vector<double> identity_experts(32, 0.0);
  for (size_t expert = 0; expert < 2; ++expert) {
    for (size_t i = 0; i < 4; ++i) {
      identity_experts[expert * 16 + i * 4 + i] = 1.0;
    }
  }

  auto lctxs = Mock2PCLink(false);
  auto run_party = [&](size_t rank) {
    Environment env(lctxs[rank]);
    silu::SecureSiluSession silu_session(lctxs[rank],
                                         static_cast<uint32_t>(rank + 1));
    auto local_tokens = rank == 0 ? tokens : zero_tokens;
    env.FeedShare("expert_choice_x", xt::adapt(local_tokens, {1, 8, 4}));

    if (rank == 0) {
      auto router_copy = router;
      auto w1_copy = identity_experts;
      auto w2_copy = identity_experts;
      env.SetPrivate("expert_choice_router", xt::adapt(router_copy, {4, 2}));
      env.SetPrivate("expert_choice_w1", xt::adapt(w1_copy, {2, 4, 4}));
      env.SetPrivate("expert_choice_w2", xt::adapt(w2_copy, {2, 4, 4}));
    } else {
      env.GetPrivate("expert_choice_router");
      env.GetPrivate("expert_choice_w1");
      env.GetPrivate("expert_choice_w2");
    }

    auto first_output = RunExpertChoiceMoeBlock(
        env.spu_ctx_.get(), env.cio_->deviceGetVar("expert_choice_x"),
        env.cio_->deviceGetVar("expert_choice_router"),
        env.cio_->deviceGetVar("expert_choice_w1"),
        env.cio_->deviceGetVar("expert_choice_w2"), config, &silu_session);
    EXPECT_TRUE(first_output.isSecret());
    EXPECT_EQ(first_output.shape(), spu::Shape({1, 8, 4}));
    auto output = RunExpertChoiceMoeBlock(
        env.spu_ctx_.get(), env.cio_->deviceGetVar("expert_choice_x"),
        env.cio_->deviceGetVar("expert_choice_router"),
        env.cio_->deviceGetVar("expert_choice_w1"),
        env.cio_->deviceGetVar("expert_choice_w2"), config, &silu_session);
    EXPECT_TRUE(output.isSecret());
    EXPECT_EQ(output.shape(), spu::Shape({1, 8, 4}));
    EXPECT_EQ(silu_session.GetInitializationCount(), 1);
    EXPECT_EQ(silu_session.GetGenerationCount(), 2);

    auto revealed = spu::kernel::hal::reveal(env.spu_ctx_.get(), output);
    auto values =
        spu::kernel::hal::dump_public_as<double>(env.spu_ctx_.get(), revealed);
    return std::vector<double>(values.begin(), values.end());
  };

  auto party0 = std::async(std::launch::async, run_party, 0);
  auto result1 = run_party(1);
  auto result0 = party0.get();

  ASSERT_EQ(result0.size(), tokens.size());
  ASSERT_EQ(result1.size(), tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    EXPECT_NEAR(result0[i], result1[i], 1e-6);
    const double expected = tokens[i] / (1.0 + std::exp(-tokens[i]));
    EXPECT_NEAR(result1[i], expected, 5e-2);
  }
}

}  // namespace
}  // namespace capricorn::vmoe
