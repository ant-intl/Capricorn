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

#include "src/adapters/vmoe/intrinsic_handler.h"

#include <vector>

#include "src/adapters/vmoe/expert_choice_moe_block.h"
#include "src/adapters/vmoe/metrics.h"
#include "src/adapters/vmoe/sparse_moe_block.h"
#include "libspu/dialect/pphlo/IR/ops.h"
#include "libspu/kernel/hal/type_cast.h"

#include "yacl/base/exception.h"

namespace capricorn::vmoe {

namespace {

constexpr std::string_view kTraceTargetPrefix = "spu.capricorn.vmoe_trace.";
constexpr std::string_view kTraceTargetSuffix = ".v1";

std::string_view ParseSafeTraceStage(std::string_view target) {
  const auto prefix = kTraceTargetPrefix;
  if (target.size() <= prefix.size() + kTraceTargetSuffix.size() ||
      target.substr(0, prefix.size()) != prefix ||
      target.substr(target.size() - kTraceTargetSuffix.size()) !=
          kTraceTargetSuffix) {
    return {};
  }
  const auto stage = target.substr(
      prefix.size(), target.size() - prefix.size() - kTraceTargetSuffix.size());
  if (stage.empty() || stage.front() == '.' || stage.back() == '.') {
    return {};
  }
  bool previous_dot = false;
  for (const char value : stage) {
    const bool dot = value == '.';
    const bool safe = dot || value == '_' || (value >= 'a' && value <= 'z') ||
                      (value >= '0' && value <= '9');
    if (!safe || (dot && previous_dot)) {
      return {};
    }
    previous_dot = dot;
  }
  return stage;
}

spu::device::OpExecutor::handler_result_t HandleVmoeIntrinsicImpl(
    spu::SPUContext* spu_ctx, mlir::Operation* op,
    absl::Span<const spu::Value> inputs, silu::SecureSiluSession* silu_session,
    const std::shared_ptr<ModelStageRecorder>& stage_recorder,
    ExpertTopTMode expert_top_t_mode) {
  YACL_ENFORCE(spu_ctx != nullptr, "SPU context must not be null");
  if (op == nullptr) {
    return std::nullopt;
  }

  auto custom_call = mlir::dyn_cast<mlir::spu::pphlo::CustomCallOp>(op);
  if (!custom_call) {
    return std::nullopt;
  }
  const auto target = custom_call.getCallTargetName().str();
  if (IsVmoeTraceTarget(target)) {
    YACL_ENFORCE(inputs.size() == 1, "{} expects one operand, got {}", target,
                 inputs.size());
    YACL_ENFORCE(op->getNumResults() == 1, "{} expects one result, got {}",
                 target, op->getNumResults());
    YACL_ENFORCE(stage_recorder != nullptr,
                 "{} requires a model stage recorder", target);
    stage_recorder->Boundary(ParseVmoeTraceStage(target));
    return std::vector<spu::Value>{inputs[0]};
  }
  if (!IsVmoeIntrinsicTarget(target)) {
    return std::nullopt;
  }

  const bool has_bias = target == kVmoeS32GeluBiasTarget ||
                        target == kVmoeS32ExpertChoiceGeluBiasTarget ||
                        target == kVmoeS32SiluBiasTarget ||
                        target == kVmoeS32ExpertChoiceSecureSiluBiasTarget;
  const size_t expected_operands = has_bias ? 6 : 4;
  YACL_ENFORCE(inputs.size() == expected_operands,
               "{} expects {} operands, got {}", target, expected_operands,
               inputs.size());
  YACL_ENFORCE(op->getNumResults() == 1, "{} expects 1 result, got {}", target,
               op->getNumResults());
  YACL_ENFORCE(spu_ctx->config().protocol() == spu::ProtocolKind::CHEETAH,
               "{} requires CHEETAH", target);
  YACL_ENFORCE(spu_ctx->getField() == spu::FieldType::FM64, "{} requires FM64",
               target);

  for (size_t index = 0; index < inputs.size(); ++index) {
    YACL_ENFORCE(inputs[index].isFxp(), "{} operand {} must be fixed point",
                 target, index);
  }
  YACL_ENFORCE(inputs[0].isSecret(), "{} token input must be secret", target);
  for (size_t index = 1; index < inputs.size(); ++index) {
    YACL_ENFORCE(inputs[index].isPrivate() && inputs[index].owner() == 1,
                 "{} model operand {} must be private to P2 (rank 1)", target,
                 index);
  }

  StageMetric intrinsic_metric(spu_ctx->lctx(), "custom_call", target);
  spu::Value output;
  if (target == kVmoeS32ExpertChoiceSecureSiluBiasTarget) {
    output = RunExpertChoiceMoeBlockWithBiasSecureSilu(
        spu_ctx, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
        inputs[5], ExpertChoiceMoeConfig::OfficialVmoeS32(), silu_session);
  } else if (target == kVmoeS32SiluBiasTarget) {
    output = RunSparseMoeBlockWithBias(
        spu_ctx, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
        inputs[5], SparseMoeConfig::OfficialVmoeS32(expert_top_t_mode),
        ExpertActivation::kSilu);
  } else if (target == kVmoeS32ExpertChoiceGeluBiasTarget) {
    output = RunExpertChoiceMoeBlockWithBias(
        spu_ctx, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
        inputs[5], ExpertChoiceMoeConfig::OfficialVmoeS32());
  } else if (target == kVmoeS32GeluBiasTarget) {
    output = RunSparseMoeBlockWithBias(spu_ctx, inputs[0], inputs[1], inputs[2],
                                       inputs[3], inputs[4], inputs[5],
                                       SparseMoeConfig::OfficialVmoeS32(
                                           expert_top_t_mode));
  } else if (target == kVmoeS32ExpertChoiceSiluTarget) {
    output = RunExpertChoiceMoeBlock(
        spu_ctx, inputs[0], inputs[1], inputs[2], inputs[3],
        ExpertChoiceMoeConfig::VmoeS32(), silu_session);
  } else {
    output = RunSparseMoeBlock(spu_ctx, inputs[0], inputs[1], inputs[2],
                               inputs[3],
                               SparseMoeConfig::VmoeS32(expert_top_t_mode));
  }
  YACL_ENFORCE(output.isSecret(), "{} output must remain secret", target);
  return std::vector<spu::Value>{std::move(output)};
}

}  // namespace

bool IsVmoeIntrinsicTarget(std::string_view target) {
  return target == kVmoeS32SiluTarget ||
         target == kVmoeS32ExpertChoiceSiluTarget ||
         target == kVmoeS32SiluBiasTarget ||
         target == kVmoeS32ExpertChoiceSecureSiluBiasTarget ||
         target == kVmoeS32GeluBiasTarget ||
         target == kVmoeS32ExpertChoiceGeluBiasTarget;
}

bool IsVmoeTraceTarget(std::string_view target) {
  return !ParseSafeTraceStage(target).empty();
}

std::string ParseVmoeTraceStage(std::string_view target) {
  YACL_ENFORCE(IsVmoeTraceTarget(target), "invalid V-MoE trace target {}",
               target);
  return std::string(ParseSafeTraceStage(target));
}

spu::device::OpExecutor::handler_result_t HandleVmoeIntrinsic(
    spu::SPUContext* spu_ctx, mlir::Operation* op,
    absl::Span<const spu::Value> inputs) {
  return HandleVmoeIntrinsicImpl(spu_ctx, op, inputs, nullptr, nullptr,
                                 ExpertTopTMode::SerialQuickSelect);
}

spu::device::OpExecutor::handler_t MakeVmoeIntrinsicHandler(
    silu::SecureSiluSession* silu_session,
    std::shared_ptr<ModelStageRecorder> stage_recorder,
    ExpertTopTMode expert_top_t_mode) {
  return [silu_session, stage_recorder = std::move(stage_recorder),
          expert_top_t_mode](
             spu::SPUContext* spu_ctx, mlir::Operation* op,
             absl::Span<const spu::Value> inputs) {
    return HandleVmoeIntrinsicImpl(spu_ctx, op, inputs, silu_session,
                                   stage_recorder, expert_top_t_mode);
  };
}

}  // namespace capricorn::vmoe
