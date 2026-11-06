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

#include "src/adapters/vmoe/sparse_moe_block.h"

#include <cmath>
#include <future>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/test_utils.h"
#include "xtensor/containers/xadapt.hpp"

namespace capricorn::vmoe {
namespace {

TEST(SparseMoeBlockTest, PublishesVersionedRealShapeContract) {
  using Signature = spu::Value (*)(spu::SPUContext*, const spu::Value&,
                                   const spu::Value&, const spu::Value&,
                                   const spu::Value&, const SparseMoeConfig&);

  static_assert(std::is_same_v<decltype(&RunSparseMoeBlock), Signature>);
  EXPECT_STREQ(kVmoeS32SiluTarget, "spu.capricorn.vmoe_s32_e8_k2_silu.v1");

  const auto config = SparseMoeConfig::VmoeS32();
  EXPECT_EQ(config.gate.m_, 50);
  EXPECT_EQ(config.gate.d_, 512);
  EXPECT_EQ(config.expert.n_, 8);
  EXPECT_EQ(config.expert.ed_, 2048);
  EXPECT_EQ(config.dispatch.k_, 2);
  EXPECT_EQ(config.dispatch.t_, 16);
}

TEST(SparseMoeBlockTest, RunsAllStagesWithOneBorrowedContext) {
  SparseMoeConfig config{
      GateRoutingConfig(/*token_nums=*/2, /*expert_nums=*/2, /*top_k=*/1,
                        /*embedding_dimension=*/2),
      DispatchConfig(/*token_nums=*/2, /*expert_nums=*/2,
                     /*activated_experts_per_token=*/1,
                     /*dispatch_expert_per_token=*/1,
                     /*embedding_dimension=*/2),
      BatchExpertComputeConfig(/*expert_nums=*/2,
                               /*token_capacity_per_expert=*/1,
                               /*embedding_dimension=*/2,
                               /*expert_dimension=*/2),
      CombineConfig(/*token_nums=*/2, /*expert_nums=*/2,
                    /*dispatch_expert_per_token=*/1,
                    /*embedding_dimension=*/2),
  };
  config.Validate();

  const std::vector<double> tokens{0.1, 0.2, 0.3, -0.1};
  const std::vector<double> zero_tokens(tokens.size(), 0.0);
  // Official router layout [embedding, expert]. Both tokens select expert 0,
  // and token 1 has the higher gate score, so capacity 1 drops token 0.
  const std::vector<double> router{1.0, -1.0, 0.5, 0.25};
  const std::vector<double> identity_experts{1.0, 0.0, 0.0, 1.0,
                                             1.0, 0.0, 0.0, 1.0};

  auto lctxs = Mock2PCLink(false);
  auto run_party = [&](size_t rank) {
    Environment env(lctxs[rank]);
    auto local_tokens = rank == 0 ? tokens : zero_tokens;
    auto xt_tokens = xt::adapt(local_tokens, {1, 2, 2});
    env.FeedShare("vmoe_x", xt_tokens);

    if (rank == 0) {
      auto router_copy = router;
      auto w1_copy = identity_experts;
      auto w2_copy = identity_experts;
      env.SetPrivate("vmoe_router", xt::adapt(router_copy, {2, 2}));
      env.SetPrivate("vmoe_w1", xt::adapt(w1_copy, {2, 2, 2}));
      env.SetPrivate("vmoe_w2", xt::adapt(w2_copy, {2, 2, 2}));
    } else {
      env.GetPrivate("vmoe_router");
      env.GetPrivate("vmoe_w1");
      env.GetPrivate("vmoe_w2");
    }

    auto output =
        RunSparseMoeBlock(env.spu_ctx_.get(), env.cio_->deviceGetVar("vmoe_x"),
                          env.cio_->deviceGetVar("vmoe_router"),
                          env.cio_->deviceGetVar("vmoe_w1"),
                          env.cio_->deviceGetVar("vmoe_w2"), config);
    EXPECT_TRUE(output.isSecret());
    EXPECT_EQ(output.shape(), spu::Shape({1, 2, 2}));

    auto revealed = spu::kernel::hal::reveal(env.spu_ctx_.get(), output);
    auto values =
        spu::kernel::hal::dump_public_as<double>(env.spu_ctx_.get(), revealed);
    return std::vector<double>(values.begin(), values.end());
  };

  auto party0 = std::async(std::launch::async, run_party, 0);
  auto result1 = run_party(1);
  auto result0 = party0.get();

  ASSERT_EQ(result0.size(), 4);
  ASSERT_EQ(result0.size(), result1.size());
  for (size_t idx = 0; idx < result0.size(); ++idx) {
    EXPECT_NEAR(result0[idx], result1[idx], 1e-6);
  }
  EXPECT_NEAR(result1[0], 0.0, 5e-2);
  EXPECT_NEAR(result1[1], 0.0, 5e-2);

  const double selected_score =
      std::exp(0.25) / (std::exp(0.25) + std::exp(-0.325));
  const double expected_0 = selected_score * (0.3 / (1.0 + std::exp(-0.3)));
  const double expected_1 = selected_score * (-0.1 / (1.0 + std::exp(0.1)));
  EXPECT_NEAR(result1[2], expected_0, 5e-2);
  EXPECT_NEAR(result1[3], expected_1, 5e-2);
}

}  // namespace
}  // namespace capricorn::vmoe
