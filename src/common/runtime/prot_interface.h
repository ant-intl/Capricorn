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

namespace capricorn {

class ProtInterface {
 public:
  ProtInterface() {}

  inline void InitEnv(std::shared_ptr<yacl::link::Context> lctx) {
    env_ptr_ = std::make_unique<Environment>(lctx);
  }

  inline void InitEnv(std::unique_ptr<Environment> env_ptr) {
    env_ptr_ = std::move(env_ptr);
  }

  inline bool IsEnvReady() { return env_ptr_ != nullptr; }

  inline std::unique_ptr<Environment> StealEnv() { return std::move(env_ptr_); }

  virtual void Compute() = 0;

  virtual ~ProtInterface() = default;

  std::unique_ptr<Environment> env_ptr_{nullptr};
};

}  // namespace capricorn