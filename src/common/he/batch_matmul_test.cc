// Copyright 2022 Ant Group Co., Ltd.
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

#include "src/common/he/batch_matmul.h"

#include <array>

#include "gtest/gtest.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

#include "yacl/utils/elapsed_timer.h"

namespace spu::mpc::cheetah {

class BatchMatMulTest
    : public ::testing::TestWithParam<std::tuple<FieldType, Shape4D, bool>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, BatchMatMulTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64),
                     testing::Values(
                         // Shape4D{64, 32, 512, 512},
                         //                Shape4D{4, 1, 2048, 768},
                         //                Shape4D{4, 18, 768, 78},
                         Shape4D{4, 1024, 16, 16}, Shape4D{64, 2, 64, 64}),
                     testing::Values(false)),
    [](const testing::TestParamInfo<BatchMatMulTest::ParamType>& p) {
      return fmt::format(
          "{}_{}x{}x{}x{}_{}", std::get<0>(p.param),
          std::get<0>(std::get<1>(p.param)), std::get<1>(std::get<1>(p.param)),
          std::get<2>(std::get<1>(p.param)), std::get<3>(std::get<1>(p.param)),
          std::get<2>(p.param) ? "Approx" : "Exact");
    });

TEST_P(BatchMatMulTest, Basic) {
  size_t kWorldSize = 2;
  auto field = std::get<0>(GetParam());
  auto dim4 = std::get<1>(GetParam());
  bool allow_approx = std::get<2>(GetParam());

  std::vector<NdArrayRef> mat(kWorldSize);

  // TODO: inputs should be multi-dim tensors.
  mat[0] = ring_rand(field, {dim4[0], dim4[1], dim4[2]});
  mat[1] = ring_rand(field, {dim4[0], dim4[2], dim4[3]});

  std::vector<NdArrayRef> result(kWorldSize);
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    int rank = lctx->Rank();
    auto matmul = std::make_shared<BatchMatMul>(lctx, rank, allow_approx);
    matmul->LazyInitKeys(field);
    SPDLOG_INFO("input elements {}", mat[rank]);
    result[rank] = matmul->BatchDotOLE(mat[rank], lctx.get(), dim4, rank == 0);
  });

  auto computed = ring_add(result[0], result[1]);

  // compute expected result
  NdArrayRef expected;
  expected = ring_zeros(field, {dim4[0], dim4[1], dim4[3]});

  for (int64_t b = 0; b < dim4[0]; b++) {
    auto lhs = mat[0]
                   .slice({b, 0, 0}, {b + 1, dim4[1], dim4[2]}, {1, 1, 1})
                   .reshape({dim4[1], dim4[2]});
    auto rhs = mat[1]
                   .slice({b, 0, 0}, {b + 1, dim4[2], dim4[3]}, {1, 1, 1})
                   .reshape({dim4[2], dim4[3]});
    auto slice = expected.slice({b, 0, 0}, {b + 1, dim4[1], dim4[3]}, {1, 1, 1})
                     .reshape({dim4[1], dim4[3]});
    ring_mmul_(slice, lhs, rhs);
  }

  EXPECT_EQ(expected.numel(), computed.numel());
  DISPATCH_ALL_FIELDS(field, [&]() {
    auto e = NdArrayView<ring2k_t>(expected);
    auto c = NdArrayView<ring2k_t>(computed);

    for (auto idx = 0; idx < expected.numel(); idx++) {
      // only exact version supported now.
      SPU_ENFORCE(e[idx] == c[idx], "expected {}, got {}, at {}", e[idx],
                  c[idx], idx);
    }
  });
}

TEST(BatchMatMulTest, PersistentSessionPreparesOnceAndServesTwoQueries) {
  constexpr size_t kWorldSize = 2;
  const auto field = FieldType::FM32;
  const Shape4D dim4{2, 2, 4, 4};

  std::array<NdArrayRef, 2> mat = {
      ring_rand(field, {dim4[0], dim4[1], dim4[2]}),
      ring_rand(field, {dim4[0], dim4[2], dim4[3]}),
  };
  std::array<std::array<NdArrayRef, 2>, 2> result;

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
    const size_t rank = lctx->Rank();
    auto session = std::make_shared<BatchMatMul>(lctx, rank == 1, false);
    session->PrepareForShape(field, dim4);
    EXPECT_TRUE(session->IsPreparedForShape(field, dim4));

    const auto stats_before = lctx->GetStats();
    const size_t sent_before = stats_before->sent_bytes.load();
    const size_t received_before = stats_before->recv_bytes.load();
    session->PrepareForShape(field, dim4);
    const auto stats_after = lctx->GetStats();
    EXPECT_EQ(stats_after->sent_bytes.load(), sent_before);
    EXPECT_EQ(stats_after->recv_bytes.load(), received_before);

    for (size_t query = 0; query < 2; ++query) {
      result[query][rank] = session->BatchDotOLE(mat[rank], dim4, rank == 0);
    }
  });

  auto expected = ring_zeros(field, {dim4[0], dim4[1], dim4[3]});
  for (int64_t batch = 0; batch < dim4[0]; ++batch) {
    auto lhs =
        mat[0]
            .slice({batch, 0, 0}, {batch + 1, dim4[1], dim4[2]}, {1, 1, 1})
            .reshape({dim4[1], dim4[2]});
    auto rhs =
        mat[1]
            .slice({batch, 0, 0}, {batch + 1, dim4[2], dim4[3]}, {1, 1, 1})
            .reshape({dim4[2], dim4[3]});
    auto output =
        expected.slice({batch, 0, 0}, {batch + 1, dim4[1], dim4[3]}, {1, 1, 1})
            .reshape({dim4[1], dim4[3]});
    ring_mmul_(output, lhs, rhs);
  }

  for (size_t query = 0; query < 2; ++query) {
    const auto computed = ring_add(result[query][0], result[query][1]);
    EXPECT_EQ(expected.numel(), computed.numel());
    DISPATCH_ALL_FIELDS(field, [&]() {
      auto expected_view = NdArrayView<ring2k_t>(expected);
      auto computed_view = NdArrayView<ring2k_t>(computed);
      for (int64_t index = 0; index < expected.numel(); ++index) {
        EXPECT_EQ(expected_view[index], computed_view[index]);
      }
    });
  }
}

}  // namespace spu::mpc::cheetah

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
