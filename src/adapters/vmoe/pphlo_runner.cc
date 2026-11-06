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

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "libspu/compiler/compile_stablehlo.h"
#include "libspu/device/api.h"
#include "libspu/device/pphlo/pphlo_executor.h"
#include "libspu/dialect/pphlo/IR/ops.h"
#include "mlir/IR/Operation.h"
#include "src/adapters/vmoe/intrinsic_handler.h"
#include "src/adapters/vmoe/metrics.h"
#include "src/capricorn/activation/secure_silu.h"

#include "yacl/base/exception.h"

namespace capricorn::vmoe {
namespace {

class TracingPphloExecutor final : public spu::device::pphlo::PPHloExecutor {
 public:
  void runKernelImpl(spu::SPUContext* sctx,
                     spu::device::SymbolScope* symbol_scope,
                     mlir::Operation& operation,
                     const spu::device::ExecutionOptions& options) override {
    std::cerr << "[capricorn-pphlo] "
              << operation.getName().getStringRef().str() << std::endl;
    spu::device::pphlo::PPHloExecutor::runKernelImpl(sctx, symbol_scope,
                                                     operation, options);
  }
};

class ProfilingPphloExecutor final : public spu::device::pphlo::PPHloExecutor {
 public:
  explicit ProfilingPphloExecutor(
      std::shared_ptr<ModelStageRecorder> stage_recorder)
      : stage_recorder_(std::move(stage_recorder)) {}

  void runKernelImpl(spu::SPUContext* sctx,
                     spu::device::SymbolScope* symbol_scope,
                     mlir::Operation& operation,
                     const spu::device::ExecutionOptions& options) override {
    bool is_trace_marker = false;
    if (auto custom_call =
            mlir::dyn_cast<mlir::spu::pphlo::CustomCallOp>(&operation)) {
      is_trace_marker =
          IsVmoeTraceTarget(custom_call.getCallTargetName().str());
    }
    if (is_trace_marker) {
      spu::device::pphlo::PPHloExecutor::runKernelImpl(sctx, symbol_scope,
                                                       operation, options);
      return;
    }

    const auto begin_stats = ReadLinkMetricSnapshot(sctx->lctx());
    const auto begin = std::chrono::steady_clock::now();
    spu::device::pphlo::PPHloExecutor::runKernelImpl(sctx, symbol_scope,
                                                     operation, options);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin);
    const auto link_delta = ReadLinkMetricSnapshot(sctx->lctx()) - begin_stats;
    stage_recorder_->RecordOperation(operation.getName().getStringRef().str(),
                                     elapsed.count(), link_delta);
  }

 private:
  std::shared_ptr<ModelStageRecorder> stage_recorder_;
};

}  // namespace

std::unique_ptr<spu::device::OpExecutor> MakeVmoePphloExecutor(
    silu::SecureSiluSession* silu_session,
    std::shared_ptr<yacl::link::Context> link,
    ExpertTopTMode expert_top_t_mode) {
  auto stage_recorder = std::make_shared<ModelStageRecorder>(std::move(link));
  std::unique_ptr<spu::device::pphlo::PPHloExecutor> executor;
  if (std::getenv("CAPRICORN_PPHLO_TRACE") != nullptr) {
    executor = std::make_unique<TracingPphloExecutor>();
  } else if (std::getenv("CAPRICORN_VMOE_PROFILE_OPERATORS") != nullptr) {
    executor = std::make_unique<ProfilingPphloExecutor>(stage_recorder);
  } else {
    executor = std::make_unique<spu::device::pphlo::PPHloExecutor>();
  }
  executor->setExtraIntrinsicHandler(
      MakeVmoeIntrinsicHandler(silu_session, stage_recorder,
                               expert_top_t_mode));
  return executor;
}

spu::ExecutableProto CompileVmoeStablehlo(
    std::string_view stablehlo, const std::vector<std::string>& input_names,
    const std::vector<spu::Visibility>& input_visibilities,
    std::string_view output_name) {
  return CompileCapricornStablehlo(stablehlo, input_names, input_visibilities,
                                   {std::string(output_name)});
}

spu::ExecutableProto CompileCapricornStablehlo(
    std::string_view stablehlo, const std::vector<std::string>& input_names,
    const std::vector<spu::Visibility>& input_visibilities,
    const std::vector<std::string>& output_names) {
  YACL_ENFORCE(!stablehlo.empty(), "StableHLO source must not be empty");
  YACL_ENFORCE(!input_names.empty(), "At least one input is required");
  YACL_ENFORCE(input_names.size() == input_visibilities.size(),
               "Input names and visibilities must have the same size: {} vs {}",
               input_names.size(), input_visibilities.size());
  YACL_ENFORCE(!output_names.empty(), "At least one output is required");

  spu::CompilerOptions options;

  spu::ExecutableProto executable;
  for (const auto& input_name : input_names) {
    YACL_ENFORCE(!input_name.empty(), "Input name must not be empty");
    executable.add_input_names(input_name);
  }
  for (const auto& output_name : output_names) {
    YACL_ENFORCE(!output_name.empty(), "Output name must not be empty");
    executable.add_output_names(output_name);
  }
  executable.set_code(spu::compiler::compileStablehloOnly(
      stablehlo, input_visibilities, options));
  return executable;
}

PersistentPphloSession::PersistentPphloSession(
    spu::SPUContext* spu_ctx, ExpertTopTMode expert_top_t_mode)
    : spu_ctx_(spu_ctx) {
  YACL_ENFORCE(spu_ctx_ != nullptr, "SPU context must not be null");
  silu_session_ = std::make_unique<silu::SecureSiluSession>(
      spu_ctx_->lctx(),
      static_cast<uint32_t>(spu_ctx_->lctx()->Rank() + 1));
  executor_ = MakeVmoePphloExecutor(silu_session_.get(), spu_ctx_->lctx(),
                                    expert_top_t_mode);
}

PersistentPphloSession::~PersistentPphloSession() = default;

void PersistentPphloSession::Execute(const spu::ExecutableProto& executable,
                                     spu::device::SymbolTable* symbols) {
  YACL_ENFORCE(symbols != nullptr, "SPU symbol table must not be null");
  spu::device::execute(executor_.get(), spu_ctx_, executable, symbols);
}

void ExecuteVmoePphlo(spu::SPUContext* spu_ctx,
                      const spu::ExecutableProto& executable,
                      spu::device::SymbolTable* symbols,
                      ExpertTopTMode expert_top_t_mode) {
  YACL_ENFORCE(spu_ctx != nullptr, "SPU context must not be null");
  YACL_ENFORCE(symbols != nullptr, "SPU symbol table must not be null");
  PersistentPphloSession session(spu_ctx, expert_top_t_mode);
  session.Execute(executable, symbols);
}

}  // namespace capricorn::vmoe
