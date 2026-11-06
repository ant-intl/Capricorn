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

#include "src/adapters/vmoe/pphlo_runner.h"

#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

namespace capricorn::vmoe {
namespace {

TEST(PphloRunnerTest, InstallsCapricornIntrinsicHandler) {
  auto executor = MakeVmoePphloExecutor();
  ASSERT_NE(executor, nullptr);
  EXPECT_TRUE(executor->getExtraIntrinsicHandler().has_value());
}

TEST(PphloRunnerTest, PublishesNonCopyablePersistentSession) {
  static_assert(!std::is_copy_constructible_v<PersistentPphloSession>);
  static_assert(!std::is_copy_assignable_v<PersistentPphloSession>);
}

TEST(PphloRunnerTest, CompilesMultipleNamedOutputs) {
  constexpr char kTwoOutputStablehlo[] = R"(
module @two_outputs {
  func.func public @main(%arg0: tensor<2xf32>)
      -> (tensor<2xf32>, tensor<2xf32>) {
    %0 = stablehlo.negate %arg0 : tensor<2xf32>
    return %arg0, %0 : tensor<2xf32>, tensor<2xf32>
  }
}
)";
  auto executable = CompileCapricornStablehlo(kTwoOutputStablehlo, {"input"},
                                              {spu::Visibility::VIS_SECRET},
                                              {"positive", "negative"});
  ASSERT_EQ(executable.output_names_size(), 2);
  EXPECT_EQ(executable.output_names(0), "positive");
  EXPECT_EQ(executable.output_names(1), "negative");
}

TEST(PphloRunnerTest, CompilesJaxStablehloIntoVersionedPphloCall) {
  constexpr char kJaxStablehlo[] = R"(
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

  const std::vector<std::string> input_names = {
      "token_embeddings", "router_weights", "expert_w1", "expert_w2"};
  const std::vector<spu::Visibility> input_visibilities = {
      spu::Visibility::VIS_SECRET, spu::Visibility::VIS_PRIVATE,
      spu::Visibility::VIS_PRIVATE, spu::Visibility::VIS_PRIVATE};
  auto executable = CompileVmoeStablehlo(kJaxStablehlo, input_names,
                                         input_visibilities, "vmoe_output");

  ASSERT_EQ(executable.input_names_size(), 4);
  for (size_t index = 0; index < input_names.size(); ++index) {
    EXPECT_EQ(executable.input_names(index), input_names[index]);
  }
  ASSERT_EQ(executable.output_names_size(), 1);
  EXPECT_EQ(executable.output_names(0), "vmoe_output");
  EXPECT_NE(executable.code().find("pphlo.custom_call"), std::string::npos);
  EXPECT_NE(executable.code().find("spu.capricorn.vmoe_s32_e8_k2_silu.v1"),
            std::string::npos);
}

TEST(PphloRunnerTest, RejectsMismatchedInputNamesAndVisibilities) {
  const std::vector<std::string> input_names = {"tokens", "weights"};
  const std::vector<spu::Visibility> input_visibilities = {
      spu::Visibility::VIS_SECRET};

  EXPECT_ANY_THROW(CompileVmoeStablehlo("module {}", input_names,
                                        input_visibilities, "vmoe_output"));
}

}  // namespace
}  // namespace capricorn::vmoe
