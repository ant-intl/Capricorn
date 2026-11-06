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

#include "src/common/runtime/environment.h"

#include "gtest/gtest.h"
#include "src/common/runtime/test_utils.h"

#include "yacl/crypto/rand/rand.h"

namespace capricorn {

TEST(Environment, BasicWork) {
  auto lctxs = Mock2PCLink(false);

  auto task0 =
      std::async([&] { return std::make_unique<Environment>(lctxs[0]); });
  auto task1 =
      std::async([&] { return std::make_unique<Environment>(lctxs[1]); });

  auto env0 = task0.get();
  auto env1 = task1.get();

  EXPECT_TRUE(env0->spu_ctx_ != nullptr);
  EXPECT_TRUE(env0->lctx_ != nullptr);
  EXPECT_TRUE(env0->spu_ctx_ != nullptr);

  EXPECT_TRUE(env1->spu_ctx_ != nullptr);
  EXPECT_TRUE(env1->lctx_ != nullptr);
  EXPECT_TRUE(env1->spu_ctx_ != nullptr);
}

}  // namespace capricorn

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
