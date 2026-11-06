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
#include "src/common/runtime/environment.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xcsv.hpp"
#include "xtensor/views/xview.hpp"

namespace capricorn {

class Environment {
 public:
  std::unique_ptr<spu::SPUContext> spu_ctx_{nullptr};
  std::shared_ptr<yacl::link::Context> lctx_{nullptr};
  std::unique_ptr<spu::device::ColocatedIo> cio_{nullptr};

  Environment(std::shared_ptr<yacl::link::Context> lctx,
              spu::ProtocolKind prot_kind = spu::ProtocolKind::CHEETAH,
              spu::FieldType field = spu::FieldType::FM64);

  ~Environment() {}

  void FeedShare(const std::string name, const spu::PtBufferView &bv);
  void SetShare(const std::string name, const spu::Value &var);

  void SetPrivate(const std::string name, const spu::PtBufferView &bv);
  void GetPrivate(const std::string name);

  void SetPublic(const std::string name, const spu::PtBufferView &bv);
};

}  // namespace capricorn
