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

#include "src/common/mpc/preprocess.h"

#include <cstdint>

#include "src/common/runtime/macro.h"

namespace capricorn::operator_ {

PreprocessResult Preprocess(
    const std::vector<std::shared_ptr<link::LinkIO>>& io_array,
    uint32_t party_id, uint32_t thread_number, uint32_t rot_number) {
  PreprocessResult result;

  // 1. Initialize ROT context
  result.rot_context =
      std::make_shared<RotContext>(party_id, thread_number, rot_number);
  result.rot_context->Init(io_array);

  // 2. Initialize BOLE context
  result.bole_context = std::make_unique<BoleContext>(
      party_id, thread_number, result.rot_context.get(),
      result.rot_context->GetIO());

  return result;
}

}  // namespace capricorn::operator_