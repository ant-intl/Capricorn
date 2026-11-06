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

#include "src/common/runtime/test_utils.h"

#include "gtest/gtest.h"

namespace capricorn {

TEST(Link, MemWork) {
  auto lctxs = Mock2PCLink(false);
  auto lctx0 = lctxs[0];
  auto lctx1 = lctxs[1];

  std::vector<uint8_t> msgs = {1, 2, 3, 4};
  yacl::Buffer buf;
  auto task0 = std::async([&]() {
    lctx0->SendAsync(lctx0->NextRank(), msgs, "mem-test");
    lctx0->WaitLinkTaskFinish();
  });
  auto task1 = std::async([&]() {
    buf = lctx1->Recv(lctx1->NextRank(), "mem-test");
    lctx1->WaitLinkTaskFinish();
  });

  task0.get();
  task1.get();

  EXPECT_EQ(yacl::ByteContainerView(msgs), yacl::ByteContainerView(buf));
}

TEST(Link, BprcWork) {
  auto lctxs = Mock2PCLink(true);
  auto lctx0 = lctxs[0];
  auto lctx1 = lctxs[1];

  std::vector<uint8_t> msgs = {1, 2, 3, 4};
  yacl::Buffer buf;
  auto task0 = std::async([&]() {
    lctx0->SendAsync(lctx0->NextRank(), msgs, "brpc-test");
    lctx0->WaitLinkTaskFinish();
  });
  auto task1 = std::async([&]() {
    buf = lctx1->Recv(lctx1->NextRank(), "brpc-test");
    lctx1->WaitLinkTaskFinish();
  });

  task0.get();
  task1.get();

  EXPECT_EQ(yacl::ByteContainerView(msgs), yacl::ByteContainerView(buf));
}

}  // namespace capricorn

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
