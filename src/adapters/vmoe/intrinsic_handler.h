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

#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "src/baselines/cryptomoe/dispatch.h"
#include "libspu/device/executor.h"

namespace capricorn::silu {
class SecureSiluSession;
}

namespace capricorn::vmoe {

class ModelStageRecorder;

bool IsVmoeIntrinsicTarget(std::string_view target);
bool IsVmoeTraceTarget(std::string_view target);
std::string ParseVmoeTraceStage(std::string_view target);

spu::device::OpExecutor::handler_result_t HandleVmoeIntrinsic(
    spu::SPUContext* spu_ctx, mlir::Operation* op,
    absl::Span<const spu::Value> inputs);

spu::device::OpExecutor::handler_t MakeVmoeIntrinsicHandler(
    silu::SecureSiluSession* silu_session,
    std::shared_ptr<ModelStageRecorder> stage_recorder = nullptr,
    ExpertTopTMode expert_top_t_mode =
        ExpertTopTMode::SerialQuickSelect);

}  // namespace capricorn::vmoe
