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

#include "libspu/core/context.h"
#include "libspu/core/value.h"

namespace capricorn::silu {
class SecureSiluSession;
}

namespace capricorn::customized {

// Convert SPU ring shares to Capricorn's 16-segment LUT SiLU protocol and
// preserve the input tensor shape. Optional sessions retain their ownership
// and support either refill mode or a prepared query budget.
spu::Value CapricornLutSiLU(spu::SPUContext* spu_ctx, const spu::Value& input,
                            uint32_t party_id,
                            silu::SecureSiluSession* session = nullptr);

spu::Value CapricornLutSiLU(
    spu::SPUContext* spu_ctx, const spu::Value& input, uint32_t party_id,
    const std::shared_ptr<silu::SecureSiluSession>& session);

}  // namespace capricorn::customized
