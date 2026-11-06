// Copyright 2026 Ant International, Ant Group. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <future>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"
#include "src/adapters/vmoe/pphlo_runner.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/test_utils.h"
#include "xtensor/containers/xadapt.hpp"

namespace capricorn::vmoe {
namespace {

constexpr char kVmoeStablehlo[] = R"(
module @jit_vmoe_block {
  func.func public @main(
      %arg0: tensor<1x50x512xf32>,
      %arg1: tensor<512x8xf32>,
      %arg2: tensor<8x512x2048xf32>,
      %arg3: tensor<8x2048x512xf32>) -> tensor<1x50x512xf32> {
    %0 = stablehlo.custom_call
        @spu.capricorn.vmoe_s32_e8_k2_silu.v1(
            %arg0, %arg1, %arg2, %arg3)
        {backend_config = "", mhlo.backend_config = {}}
        : (tensor<1x50x512xf32>, tensor<512x8xf32>,
           tensor<8x512x2048xf32>, tensor<8x2048x512xf32>)
        -> tensor<1x50x512xf32>
    return %0 : tensor<1x50x512xf32>
  }
}
)";

TEST(VmoeSingleBlockE2eTest, ExecutesRealShapeWithoutIntermediateReveal) {
  const std::vector<std::string> input_names = {
      "token_embeddings", "router_weights", "expert_w1", "expert_w2"};
  const std::vector<spu::Visibility> input_visibilities = {
      spu::Visibility::VIS_SECRET, spu::Visibility::VIS_PRIVATE,
      spu::Visibility::VIS_PRIVATE, spu::Visibility::VIS_PRIVATE};
  const auto executable = CompileVmoeStablehlo(
      kVmoeStablehlo, input_names, input_visibilities, "vmoe_output");

  auto link_contexts = Mock2PCLink(false);
  auto run_party = [&](size_t rank) {
    Environment environment(link_contexts[rank]);

    std::vector<float> token_share(1 * 50 * 512, 0.0F);
    auto token_tensor = xt::adapt(token_share, {1, 50, 512});
    environment.FeedShare(input_names[0], token_tensor);

    if (rank == 1) {
      std::vector<float> router_weights(512 * 8, 0.0F);
      std::vector<float> expert_w1(8 * 512 * 2048, 0.0F);
      std::vector<float> expert_w2(8 * 2048 * 512, 0.0F);
      auto router_tensor = xt::adapt(router_weights, {512, 8});
      auto expert_w1_tensor = xt::adapt(expert_w1, {8, 512, 2048});
      auto expert_w2_tensor = xt::adapt(expert_w2, {8, 2048, 512});
      environment.SetPrivate(input_names[1], router_tensor);
      environment.SetPrivate(input_names[2], expert_w1_tensor);
      environment.SetPrivate(input_names[3], expert_w2_tensor);
    } else {
      environment.GetPrivate(input_names[1]);
      environment.GetPrivate(input_names[2]);
      environment.GetPrivate(input_names[3]);
    }

    ExecuteVmoePphlo(environment.spu_ctx_.get(), executable,
                     &environment.cio_->deviceSymbols());
    auto output = environment.cio_->deviceGetVar("vmoe_output");
    EXPECT_TRUE(output.isSecret());
    EXPECT_EQ(output.shape(), spu::Shape({1, 50, 512}));

    auto revealed =
        spu::kernel::hal::reveal(environment.spu_ctx_.get(), output);
    auto values = spu::kernel::hal::dump_public_as<double>(
        environment.spu_ctx_.get(), revealed);
    return std::vector<double>(values.begin(), values.end());
  };

  auto party_zero = std::async(std::launch::async, run_party, 0);
  auto party_one_result = run_party(1);
  auto party_zero_result = party_zero.get();

  ASSERT_EQ(party_zero_result.size(), 1 * 50 * 512);
  ASSERT_EQ(party_zero_result.size(), party_one_result.size());
  for (size_t index = 0; index < party_zero_result.size(); ++index) {
    EXPECT_NEAR(party_zero_result[index], party_one_result[index], 1e-6);
    EXPECT_NEAR(party_one_result[index], 0.0, 5e-2);
  }
}

}  // namespace
}  // namespace capricorn::vmoe
