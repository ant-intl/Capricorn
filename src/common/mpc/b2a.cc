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

#include "src/common/mpc/b2a.h"

#include <algorithm>

namespace capricorn::operator_ {

B2AContext::B2AContext(const uint32_t party_id, RotContext* rot_context,
                       const uint32_t thread_num)
    : party_id_(party_id),
      thread_num_(
          std::min(thread_num, std::thread::hardware_concurrency() - 1)) {
  mux_context_ =
      std::make_unique<MuxContext>(party_id_, rot_context, thread_num_);
}

void B2AContext::B2A(uint32_t n, const uint8_t* input_choice,
                     uint64_t* output_z, uint32_t out_bw) {
  // B2A is essentially MUX with value = 1
  // output = choice ? 1 : 0
  mux_context_->Mux(n, input_choice, static_cast<uint64_t>(1), output_z,
                    out_bw);
}

}  // namespace capricorn::operator_