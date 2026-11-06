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

#include "src/adapters/vmoe/config.h"

#include "gtest/gtest.h"

namespace capricorn::vmoe {
namespace {

TEST(SparseMoeConfigTest, BuildsVmoeS32Contract) {
  const auto config = SparseMoeConfig::VmoeS32();

  EXPECT_EQ(config.gate.m_, 50);
  EXPECT_EQ(config.gate.n_, 8);
  EXPECT_EQ(config.gate.k_, 2);
  EXPECT_EQ(config.gate.d_, 512);

  EXPECT_EQ(config.dispatch.m_, 50);
  EXPECT_EQ(config.dispatch.n_, 8);
  EXPECT_EQ(config.dispatch.k_, 2);
  EXPECT_EQ(config.dispatch.t_, 16);

  EXPECT_EQ(config.expert.n_, 8);
  EXPECT_EQ(config.expert.t_, 16);
  EXPECT_EQ(config.expert.d_, 512);
  EXPECT_EQ(config.expert.ed_, 2048);

  EXPECT_EQ(config.combine.m_, 50);
  EXPECT_EQ(config.combine.n_, 8);
  EXPECT_EQ(config.combine.t_, 16);
  EXPECT_EQ(config.combine.d_, 512);
  EXPECT_NO_THROW(config.Validate());
}

TEST(SparseMoeConfigTest, RejectsInconsistentModuleDimensions) {
  auto config = SparseMoeConfig::VmoeS32();
  config.expert.d_ = 256;

  EXPECT_ANY_THROW(config.Validate());
}

TEST(SparseMoeConfigTest, BuildsOfficialEvalCapacityContract) {
  const auto config = SparseMoeConfig::OfficialVmoeS32();

  EXPECT_EQ(config.dispatch.t_, 52);
  EXPECT_EQ(config.expert.t_, 52);
  EXPECT_EQ(config.combine.t_, 52);
  EXPECT_NO_THROW(config.Validate());
}

TEST(SparseMoeConfigTest, SelectsParallelQuickSelectExplicitly) {
  const auto config = SparseMoeConfig::OfficialVmoeS32(
      ExpertTopTMode::ParallelQuickSelect);

  EXPECT_EQ(config.dispatch.top_t_mode_,
            ExpertTopTMode::ParallelQuickSelect);
  EXPECT_EQ(config.dispatch.t_, 52);
  EXPECT_NO_THROW(config.Validate());
}

TEST(ExpertChoiceMoeConfigTest, DefinesRealShapeS32Contract) {
  const auto config = ExpertChoiceMoeConfig::VmoeS32();

  EXPECT_EQ(config.routing.m_, 50);
  EXPECT_EQ(config.routing.n_, 8);
  EXPECT_EQ(config.routing.t_, 16);
  EXPECT_EQ(config.routing.d_, 512);

  EXPECT_EQ(config.expert.n_, 8);
  EXPECT_EQ(config.expert.t_, 16);
  EXPECT_EQ(config.expert.d_, 512);
  EXPECT_EQ(config.expert.ed_, 2048);

  EXPECT_EQ(config.combine.m_, 50);
  EXPECT_EQ(config.combine.n_, 8);
  EXPECT_EQ(config.combine.t_, 16);
  EXPECT_EQ(config.combine.d_, 512);
  EXPECT_NO_THROW(config.Validate());
}

TEST(ExpertChoiceMoeConfigTest, RejectsCapacityMismatch) {
  auto config = ExpertChoiceMoeConfig::VmoeS32();
  config.combine.t_ = 15;

  EXPECT_ANY_THROW(config.Validate());
}

TEST(ExpertChoiceMoeConfigTest, BuildsOfficialComparisonCapacityContract) {
  const auto config = ExpertChoiceMoeConfig::OfficialVmoeS32();

  EXPECT_EQ(config.routing.t_, 28);
  EXPECT_EQ(config.expert.t_, 28);
  EXPECT_EQ(config.combine.t_, 28);
  EXPECT_NO_THROW(config.Validate());
}

}  // namespace
}  // namespace capricorn::vmoe
