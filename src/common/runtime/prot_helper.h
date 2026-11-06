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
#include <unordered_map>

#include "libspu/core/context.h"
#include "libspu/core/pt_buffer_view.h"
#include "libspu/core/value.h"
#include "libspu/device/io.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/prot_interface.h"

namespace capricorn {

#define REGISTER_PROT(PROT_NAME)                          \
  std::unique_ptr<ProtInterface> inline MakeProtocol(     \
      const PROT_NAME##Config& config, bool is_server) {  \
    if (is_server) {                                      \
      return std::make_unique<PROT_NAME##Server>(config); \
    }                                                     \
    return std::make_unique<PROT_NAME##User>(config);     \
  }

}  // namespace capricorn
