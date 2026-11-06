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
#include "src/common/selection/customized_sort.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "libspu/core/bit_utils.h"
#include "libspu/core/context.h"
#include "libspu/core/trace.h"
#include "libspu/core/type.h"
#include "libspu/core/vectorize.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/random.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hal/utils.h"
#include "libspu/kernel/hlo/shuffle.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "src/common/runtime/macro.h"

namespace capricorn::customized {

using spu::kernel::hal::CompFn;

namespace {

struct QuickSelectState {
  int64_t lo;
  int64_t hi;
  bool done;
};

void SwapValues(absl::Span<spu::Value> values, const spu::Index &lhs_indices,
                const spu::Index &rhs_indices) {
  YACL_ENFORCE(lhs_indices.size() == rhs_indices.size());
  if (lhs_indices.empty()) {
    return;
  }

  for (auto &value : values) {
    auto lhs = value.data().linear_gather(lhs_indices);
    auto rhs = value.data().linear_gather(rhs_indices);
    value.data().linear_scatter(lhs, rhs_indices);
    value.data().linear_scatter(rhs, lhs_indices);
  }
}

std::vector<spu::Value> BuildShuffle(spu::SPUContext *ctx,
                                     const spu::Value &input,
                                     const std::vector<spu::Value> &payloads) {
  const auto &shape = input.shape();
  const int64_t rows = shape[0];
  const int64_t width = shape[1];
  for (const auto &payload : payloads) {
    SPU_ENFORCE(payload.shape() == shape,
                "payload shape {} must match input shape {}", payload.shape(),
                shape);
  }

  const auto index_dtype =
      ctx->config().field() == spu::FieldType::FM32 ? spu::DT_I32 : spu::DT_I64;
  auto public_indices = spu::kernel::hal::iota(ctx, index_dtype, width);
  auto index_matrix = spu::kernel::hal::broadcast_to(
      ctx, spu::kernel::hal::reshape(ctx, public_indices, {1, width}),
      {rows, width});
  auto secret_index_matrix =
      spu::kernel::hal::_p2s(ctx, index_matrix).setDtype(index_dtype);

  SPU_ENFORCE(ctx->hasKernel("rand_perm_m") && ctx->hasKernel("perm_am"),
              "Cheetah native permutation kernels are required; refusing "
              "SPU Shuffle's random-key SimpleSort fallback");
  SPDLOG_INFO(
      "BatchedQuickSelectTopK: using SPU HLO Shuffle native permutation "
      "path along candidate axis");
  std::vector<spu::Value> shuffle_inputs{input, secret_index_matrix};
  shuffle_inputs.reserve(2 + payloads.size());
  shuffle_inputs.insert(shuffle_inputs.end(), payloads.begin(), payloads.end());
  auto shuffled = spu::kernel::hlo::Shuffle(ctx, shuffle_inputs, 1);
  SPU_ENFORCE(shuffled.size() == shuffle_inputs.size());
  shuffled[0].setDtype(input.dtype());
  shuffled[1].setDtype(index_dtype);
  for (size_t i = 0; i < payloads.size(); ++i) {
    shuffled[i + 2].setDtype(payloads[i].dtype());
  }
  for (auto &value : shuffled) {
    value = spu::kernel::hal::reshape(ctx, value, {rows * width});
  }
  return shuffled;
}

}  // namespace

std::pair<spu::Value, spu::Value> _conditional_select(spu::SPUContext *ctx,
                                                      const spu::Value &cond,
                                                      const spu::Value &lhs,
                                                      const spu::Value &rhs) {
  if (lhs.storage_type().isa<spu::BShare>() &&
      rhs.storage_type().isa<spu::BShare>()) {
    // SPDLOG_INFO("BShare in conditional select");
    auto first =
        spu::kernel::hal::_xor(
            ctx, lhs,
            spu::kernel::hal::mul(
                ctx, cond,
                spu::kernel::hal::_xor(ctx, lhs, rhs).setDtype(lhs.dtype())))
            .setDtype(lhs.dtype());
    auto second = spu::kernel::hal::_xor(
                      ctx, spu::kernel::hal::_xor(ctx, lhs, rhs), first)
                      .setDtype(lhs.dtype());
    return std::make_pair(first, second);
  }

  auto first = spu::kernel::hal::add(
      ctx, lhs,
      spu::kernel::hal::mul(ctx, cond, spu::kernel::hal::sub(ctx, rhs, lhs)));
  auto second =
      spu::kernel::hal::sub(ctx, spu::kernel::hal::add(ctx, lhs, rhs), first);

  return std::make_pair(first, second);
}

void _conditional_swap(spu::SPUContext *ctx, const spu::Value &cond,
                       absl::Span<spu::Value> arr,
                       const spu::Index &lhs_indices,
                       const spu::Index &rhs_indices) {
  if (lhs_indices.empty() ||
      (lhs_indices.size() == 1 && rhs_indices.size() == 1 &&
       lhs_indices[0] == rhs_indices[0])) {
    return;
  }

  const auto num_operands = arr.size();

  for (size_t i = 0; i < num_operands; ++i) {
    auto lhs_arr =
        spu::Value(arr[i].data().linear_gather(lhs_indices), arr[i].dtype());
    auto rhs_arr =
        spu::Value(arr[i].data().linear_gather(rhs_indices), arr[i].dtype());

    auto [first, second] = _conditional_select(ctx, cond, lhs_arr, rhs_arr);

    arr[i].data().linear_scatter(first.data(), rhs_indices);
    arr[i].data().linear_scatter(second.data(), lhs_indices);
  }
}

std::vector<spu::Value> flatten_odd_even_merge_sort(
    spu::SPUContext *ctx, const CompFn &comparator_body, const spu::Value &key,
    const std::vector<spu::Value> &payloads, int64_t width) {
  YACL_ENFORCE(key.numel() % width == 0);
  for (size_t k = 0; k < payloads.size(); ++k) {
    YACL_ENFORCE(payloads[k].numel() % width == 0);
  }

  // make a copy for inplace sort
  std::vector<spu::Value> ret = {key.clone()};
  for (size_t k = 0; k < payloads.size(); ++k) {
    if (payloads[k].storage_type().isa<spu::Public>()) {
      SPDLOG_INFO("Force to convert into BShare");
      COMM_TIMER_START(ctx->lctx(), CAPRICORN_ONE_HOT_BSHARE);
      auto bshare_payload =
          spu::mpc::p2b(ctx, payloads[k]).setDtype(payloads[k].dtype());
      COMM_TIMER_END_PRINT(ctx->lctx(), CAPRICORN_ONE_HOT_BSHARE);
      ret.push_back(std::move(bshare_payload));
    } else {
      ret.push_back(payloads[k].clone());
    }
  }

  // sort by per network layer for memory optimizations, sorting N elements
  // needs log2(N) stages, and the i_th stage has i layers, which means the
  // same latency cost as BitonicSort but less _cmp_swap unit.
  const int64_t n = width;
  const int64_t m = key.numel() / n;
  for (int64_t max_gap_in_stage = 1; max_gap_in_stage < n;
       max_gap_in_stage += max_gap_in_stage) {
    for (int64_t step = max_gap_in_stage; step > 0; step /= 2) {
      // collect index pairs that can be computed parallelly.
      spu::Index lhs_indices;
      spu::Index rhs_indices;

      for (int64_t j = step % max_gap_in_stage; j + step < n;
           j += step + step) {
        for (int64_t i = 0; i < step; i++) {
          auto lhs_idx = i + j;
          auto rhs_idx = i + j + step;

          if (rhs_idx >= n) break;

          auto range = max_gap_in_stage + max_gap_in_stage;
          if (lhs_idx / range == rhs_idx / range) {
            lhs_indices.emplace_back(lhs_idx);
            rhs_indices.emplace_back(rhs_idx);
          }
        }
      }

      spu::Index extend_lhs_indices;
      spu::Index extend_rhs_indices;
      for (int64_t _m = 0; _m < m; ++_m) {
        int64_t offset = _m * width;
        for (size_t i = 0; i < lhs_indices.size(); ++i) {
          extend_lhs_indices.emplace_back(lhs_indices[i] + offset);
          extend_rhs_indices.emplace_back(rhs_indices[i] + offset);
        }
      }

      std::vector<spu::Value> values(2);

      values[0] = spu::Value(ret[0].data().linear_gather(extend_lhs_indices),
                             ret[0].dtype());
      values[1] = spu::Value(ret[0].data().linear_gather(extend_rhs_indices),
                             ret[0].dtype());

      auto predicate = comparator_body(values);

      _conditional_swap(ctx, predicate, absl::MakeSpan(ret), extend_lhs_indices,
                        extend_rhs_indices);
    }
  }

  return ret;
}

std::vector<spu::Value> sort_2d(spu::SPUContext *ctx, const spu::Value &key,
                                const std::vector<spu::Value> &payloads,
                                SortDirection direction) {
  CompFn comp_fn =
      [ctx, direction](absl::Span<const spu::Value> values) -> spu::Value {
    // single key with extra payload
    YACL_ENFORCE(values.size() == 2);
    if (direction == SortDirection::Ascending) {
      return spu::kernel::hal::less(ctx, values[0], values[1]);
    }
    return spu::kernel::hal::greater(ctx, values[0], values[1]);
  };

  auto shape2d = key.shape();
  int64_t width = shape2d[1];
  auto flatten_key = spu::kernel::hal::reshape(ctx, key, {key.numel()});
  std::vector<spu::Value> flaten_payloads;
  for (size_t k = 0; k < payloads.size(); ++k) {
    flaten_payloads.push_back(
        spu::kernel::hal::reshape(ctx, payloads[k], {payloads[k].numel()}));
  }
  auto flatten_result = flatten_odd_even_merge_sort(ctx, comp_fn, flatten_key,
                                                    flaten_payloads, width);

  std::vector<spu::Value> result;
  for (size_t k = 0; k < flatten_result.size(); ++k) {
    result.push_back(
        spu::kernel::hal::reshape(ctx, flatten_result[k], shape2d));
  }
  return result;
}

std::vector<spu::Value> topk_2d(spu::SPUContext *ctx, const spu::Value &key,
                                const std::vector<spu::Value> &payloads,
                                int64_t k) {
  auto shape2d = key.shape();
  int64_t d1 = shape2d[0];
  int64_t d2 = shape2d[1];
  SPU_ENFORCE(k <= d2, "k must be less than or equal to d2");
  auto sort_result = sort_2d(ctx, key, payloads, SortDirection::Descending);

  auto _key = sort_result[0];
  auto _payload = sort_result[1];

  std::vector<spu::Value> ret;
  for (int64_t k = 0; k < int64_t(sort_result.size()); k++) {
    auto _cur = sort_result[k];
    std::vector<spu::Value> vec_cur;
    for (int64_t i = 0; i < d1; i++) {
      vec_cur.push_back(
          spu::kernel::hal::slice(ctx, _cur, {i, 0}, {i + 1, k}, {}));
    }
    ret.push_back(spu::kernel::hal::concatenate(ctx, vec_cur, 0));
  }
  return ret;
}

std::vector<spu::Value> batched_quick_select_topk_2d(
    spu::SPUContext *ctx, const spu::Value &input,
    const std::vector<spu::Value> &payloads, int64_t k) {
  SPU_ENFORCE(input.shape().ndim() == 2, "input must be 2-D, got {} dimensions",
              input.shape().ndim());
  SPU_ENFORCE(!input.isPublic(), "input must be secret");

  const int64_t rows = input.shape()[0];
  const int64_t width = input.shape()[1];
  SPU_ENFORCE(rows > 0 && width > 0, "input must be non-empty");
  SPU_ENFORCE(k > 0 && k <= width, "k={} must be in the range [1, {}]", k,
              width);
  auto values = BuildShuffle(ctx, input, payloads);
  std::vector<QuickSelectState> states(
      rows, QuickSelectState{/*lo=*/0, /*hi=*/width - 1, /*done=*/false});
  int64_t unfinished = rows;
  int64_t rounds = 0;

  while (unfinished > 0) {
    int64_t padded_width = 0;
    for (auto &state : states) {
      if (state.done) {
        continue;
      }
      if (state.hi <= state.lo) {
        state.done = true;
        --unfinished;
        continue;
      }
      padded_width =
          std::max(padded_width,
                   state.hi == state.lo + 1 ? int64_t{1} : state.hi - state.lo);
    }
    if (unfinished == 0) {
      break;
    }
    SPU_ENFORCE(padded_width > 0);

    // Every row contributes padded_width comparisons. Completed rows and
    // unused tail slots compare an element with itself and are ignored after
    // reveal, so all active partitions share one MPC comparison/reveal round.
    spu::Index lhs_indices;
    spu::Index rhs_indices;
    lhs_indices.reserve(rows * padded_width);
    rhs_indices.reserve(rows * padded_width);
    for (int64_t row = 0; row < rows; ++row) {
      const auto &state = states[row];
      const int64_t row_offset = row * width;
      const int64_t safe_lo = state.done ? 0 : state.lo;
      const int64_t valid_width =
          state.done
              ? 0
              : (state.hi == state.lo + 1 ? int64_t{1} : state.hi - state.lo);
      for (int64_t pos = 0; pos < padded_width; ++pos) {
        lhs_indices.emplace_back(row_offset + safe_lo);
        rhs_indices.emplace_back(
            row_offset + (pos < valid_width ? safe_lo + pos + 1 : safe_lo));
      }
    }

    auto lhs = spu::Value(values[0].data().linear_gather(lhs_indices),
                          values[0].dtype());
    auto rhs = spu::Value(values[0].data().linear_gather(rhs_indices),
                          values[0].dtype());
    auto predicates = spu::kernel::hal::greater(ctx, lhs, rhs);
    auto public_predicates = spu::kernel::hal::dump_public_as<bool>(
        ctx, spu::kernel::hal::reveal(ctx, predicates));

    spu::Index partition_lhs;
    spu::Index partition_rhs;
    spu::Index pivot_lhs;
    spu::Index pivot_rhs;

    for (int64_t row = 0; row < rows; ++row) {
      auto &state = states[row];
      if (state.done) {
        continue;
      }

      const int64_t row_offset = row * width;
      const int64_t predicate_offset = row * padded_width;
      if (state.hi == state.lo + 1) {
        // Match SPU HandleSmallArray/CompSwapSingle exactly.
        if (!public_predicates[predicate_offset]) {
          pivot_lhs.emplace_back(row_offset + state.lo);
          pivot_rhs.emplace_back(row_offset + state.hi);
        }
        state.done = true;
        --unfinished;
        continue;
      }

      int64_t left = state.lo + 1;
      int64_t right = state.hi;
      const int64_t first_candidate = left;
      for (;;) {
        while (right >= left &&
               !public_predicates[predicate_offset + left - first_candidate]) {
          ++left;
        }
        while (right >= left &&
               public_predicates[predicate_offset + right - first_candidate]) {
          --right;
        }
        if (right < left) {
          break;
        }
        partition_lhs.emplace_back(row_offset + left);
        partition_rhs.emplace_back(row_offset + right);
        ++left;
        --right;
      }

      pivot_lhs.emplace_back(row_offset + state.lo);
      pivot_rhs.emplace_back(row_offset + right);

      // k is 1-based in SPU's quick-select state transition.
      if (right >= k - 1) {
        state.hi = right - 1;
      }
      if (right <= k - 1) {
        state.lo = left;
      }
    }

    SwapValues(absl::MakeSpan(values), partition_lhs, partition_rhs);
    SwapValues(absl::MakeSpan(values), pivot_lhs, pivot_rhs);

    ++rounds;
    SPU_ENFORCE(rounds <= width,
                "batched quick-select did not converge within {} rounds",
                width);
  }

  spu::Index topk_indices;
  topk_indices.reserve(rows * k);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < k; ++column) {
      topk_indices.emplace_back(row * width + column);
    }
  }

  std::vector<spu::Value> result;
  result.reserve(values.size());
  for (const auto &value : values) {
    result.emplace_back(spu::kernel::hal::reshape(
        ctx,
        spu::Value(value.data().linear_gather(topk_indices), value.dtype()),
        {rows, k}));
  }

  SPDLOG_INFO(
      "BatchedQuickSelectTopK: rows={}, width={}, k={}, partition rounds={}",
      rows, width, k, rounds);
  return result;
}

std::vector<spu::Value> batched_quick_select_topk_2d(spu::SPUContext *ctx,
                                                     const spu::Value &input,
                                                     int64_t k) {
  return batched_quick_select_topk_2d(ctx, input, {}, k);
}

}  // namespace capricorn::customized
