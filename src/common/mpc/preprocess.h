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

#include <cstdint>
#include <memory>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "src/common/mpc/bole_context.h"
#include "src/common/mpc/rot_context.h"
#include "src/common/runtime/link/link_io.h"

namespace capricorn::operator_ {
struct PreprocessResult {
  /// ROT context for OT operations
  std::shared_ptr<RotContext> rot_context;

  /// BOLE context for homomorphic multiplication
  std::unique_ptr<BoleContext> bole_context;
};

PreprocessResult Preprocess(
    const std::vector<std::shared_ptr<link::LinkIO>>& io_array,
    uint32_t party_id, uint32_t thread_number, uint32_t rot_number = 0);

}  // namespace capricorn::operator_