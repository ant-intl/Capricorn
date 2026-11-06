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

#include "src/common/nn/softmax.h"

#include <vector>

#include "absl/types/span.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/basic_unary.h"
#include "libspu/kernel/hlo/reduce.h"
#include "spdlog/spdlog.h"

namespace capricorn::customized {
namespace {

spu::Value ReduceMax_2d(spu::SPUContext* ctx, const spu::Value& input,
                        int64_t axis) {
  SPDLOG_INFO("ReduceMax: input shape {}, axis {}", input.shape(), axis);

  // Use TreeReduce with hal::max as the reducer
  auto result = spu::kernel::hlo::TreeReduce(
      ctx, {input}, axis,
      [&](absl::Span<const spu::Value> lhs, absl::Span<const spu::Value> rhs) {
        return std::vector<spu::Value>{
            spu::kernel::hal::max(ctx, lhs[0], rhs[0])};
      });

  SPDLOG_INFO("ReduceMax: result shape {}", result[0].shape());
  return result[0];
}

spu::Value ReduceSum_2d(spu::SPUContext* ctx, const spu::Value& input,
                        int64_t axis) {
  SPDLOG_INFO("ReduceSum: input shape {}, axis {}", input.shape(), axis);

  // Use TreeReduce with hal::add as the reducer
  auto result = spu::kernel::hlo::TreeReduce(
      ctx, {input}, axis,
      [&](absl::Span<const spu::Value> lhs, absl::Span<const spu::Value> rhs) {
        return std::vector<spu::Value>{
            spu::kernel::hal::add(ctx, lhs[0], rhs[0])};
      });

  SPDLOG_INFO("ReduceSum: result shape {}", result[0].shape());
  return result[0];
}

}  // namespace

spu::Value Softmax_2d(spu::SPUContext* spu_ctx, const spu::Value& input,
                      int64_t axis) {
  SPDLOG_INFO("Softmax: input shape {}", input.shape());

  // Step 1: Compute max over the specified axis for numerical stability
  // max_val = max(input, axis=axis, keepdims=true)
  SPDLOG_INFO("Softmax: computing max");
  auto max_val = ReduceMax_2d(spu_ctx, input, axis);
  SPDLOG_INFO("Softmax: max_val shape {}", max_val.shape());

  // Step 2: Broadcast max_val to match input shape for subtraction
  SPDLOG_INFO("Softmax: broadcasting max_val");
  auto max_val_broadcasted =
      spu::kernel::hal::broadcast_to(spu_ctx, max_val, input.shape());
  SPDLOG_INFO("Softmax: max_val_broadcasted shape {}",
              max_val_broadcasted.shape());

  // Step 3: Subtract max (broadcasted) from input
  // shifted = input - max_val
  SPDLOG_INFO("Softmax: subtracting max");
  auto shifted = spu::kernel::hlo::Sub(spu_ctx, input, max_val_broadcasted);

  // Step 4: Compute exp(shifted)
  // exp_shifted = exp(shifted)
  SPDLOG_INFO("Softmax: computing exp");
  auto exp_shifted = spu::kernel::hlo::Exp(spu_ctx, shifted);
  SPDLOG_INFO("Softmax: exp_shifted shape {}", exp_shifted.shape());

  // Step 5: Compute sum of exp over the specified axis
  // sum_exp = sum(exp_shifted, axis=axis, keepdims=true)
  SPDLOG_INFO("Softmax: computing sum");
  auto sum_exp = ReduceSum_2d(spu_ctx, exp_shifted, axis);
  SPDLOG_INFO("Softmax: sum_exp shape {}", sum_exp.shape());

  // Step 6: Broadcast sum_exp to match exp_shifted shape for division
  SPDLOG_INFO("Softmax: broadcasting sum_exp");
  auto sum_exp_broadcasted =
      spu::kernel::hal::broadcast_to(spu_ctx, sum_exp, exp_shifted.shape());
  SPDLOG_INFO("Softmax: sum_exp_broadcasted shape {}",
              sum_exp_broadcasted.shape());

  // Step 7: Divide exp by sum
  // output = exp_shifted / sum_exp
  SPDLOG_INFO("Softmax: dividing");
  auto output =
      spu::kernel::hlo::Div(spu_ctx, exp_shifted, sum_exp_broadcasted);
  SPDLOG_INFO("Softmax: output shape {}", output.shape());

  return output;
}

}  // namespace capricorn::customized
