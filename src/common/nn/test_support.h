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
#include <future>
#include <memory>
#include <vector>

#include "yacl/link/context.h"

namespace capricorn::test {

struct MatrixTestParams {
  int64_t row;
  int64_t col;
};

inline void FinishTestLinks(
    const std::vector<std::shared_ptr<yacl::link::Context>>& links) {
  // Spawned SiLU/SPU contexts share these root transport channels. Drain both
  // parties concurrently before either BRPC receiver is destroyed; completed
  // numeric results alone do not imply all asynchronous sends were delivered.
  std::vector<std::future<void>> finish_tasks;
  finish_tasks.reserve(links.size());
  for (const auto& link : links) {
    finish_tasks.emplace_back(std::async(
        std::launch::async, [link]() { link->WaitLinkTaskFinish(); }));
  }
  for (auto& task : finish_tasks) {
    task.get();
  }
}

}  // namespace capricorn::test
