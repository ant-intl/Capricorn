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
#include "libspu/kernel/hal/permute.h"        // bad reference
#include "libspu/kernel/hal/prot_wrapper.h"   // bad reference
#include "libspu/kernel/hal/public_helper.h"  // bad reference

namespace capricorn::customized {

using spu::kernel::hal::SortDirection;

std::vector<spu::Value> sort_2d(
    spu::SPUContext* ctx, const spu::Value& key,
    const std::vector<spu::Value>& payloads,
    SortDirection direction = SortDirection::Descending);

std::vector<spu::Value> topk_2d(spu::SPUContext* ctx, const spu::Value& key,
                                const std::vector<spu::Value>& payloads,
                                int64_t k);

// Runs the same randomized quick-select used by SPU TopK on every row of a
// 2-D secret tensor. Partition comparisons from all rows are issued together;
// in each round, shorter active intervals are padded to the longest interval
// in that round. Returns top-k values and their original column indices.
std::vector<spu::Value> batched_quick_select_topk_2d(spu::SPUContext* ctx,
                                                     const spu::Value& input,
                                                     int64_t k);

// Same QuickSelect, with payloads shuffled and partitioned in lockstep with
// input. Returns {top-k keys, original column indices, top-k payloads...}.
std::vector<spu::Value> batched_quick_select_topk_2d(
    spu::SPUContext* ctx, const spu::Value& input,
    const std::vector<spu::Value>& payloads, int64_t k);

}  // namespace capricorn::customized
