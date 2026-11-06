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
#include <vector>

#include "src/baselines/cryptomoe/dispatch.h"
#include "libspu/device/executor.h"
#include "libspu/device/symbol_table.h"

#include "yacl/link/context.h"

#include "libspu/spu.pb.h"

namespace capricorn::silu {
class SecureSiluSession;
}

namespace capricorn::vmoe {

std::unique_ptr<spu::device::OpExecutor> MakeVmoePphloExecutor(
    silu::SecureSiluSession* silu_session = nullptr,
    std::shared_ptr<yacl::link::Context> link = nullptr,
    ExpertTopTMode expert_top_t_mode =
        ExpertTopTMode::SerialQuickSelect);

class PersistentPphloSession {
 public:
  explicit PersistentPphloSession(
      spu::SPUContext* spu_ctx,
      ExpertTopTMode expert_top_t_mode =
          ExpertTopTMode::SerialQuickSelect);
  ~PersistentPphloSession();

  PersistentPphloSession(const PersistentPphloSession&) = delete;
  PersistentPphloSession& operator=(const PersistentPphloSession&) = delete;

  void Execute(const spu::ExecutableProto& executable,
               spu::device::SymbolTable* symbols);

 private:
  spu::SPUContext* spu_ctx_;
  std::unique_ptr<silu::SecureSiluSession> silu_session_;
  std::unique_ptr<spu::device::OpExecutor> executor_;
};

spu::ExecutableProto CompileCapricornStablehlo(
    std::string_view stablehlo, const std::vector<std::string>& input_names,
    const std::vector<spu::Visibility>& input_visibilities,
    const std::vector<std::string>& output_names);

spu::ExecutableProto CompileVmoeStablehlo(
    std::string_view stablehlo, const std::vector<std::string>& input_names,
    const std::vector<spu::Visibility>& input_visibilities,
    std::string_view output_name);

void ExecuteVmoePphlo(spu::SPUContext* spu_ctx,
                      const spu::ExecutableProto& executable,
                      spu::device::SymbolTable* symbols,
                      ExpertTopTMode expert_top_t_mode =
                          ExpertTopTMode::SerialQuickSelect);

}  // namespace capricorn::vmoe
