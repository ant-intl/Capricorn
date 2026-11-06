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

#include "src/common/mpc/mux.h"

namespace capricorn::operator_ {

/**
 * @brief B2A (Boolean to Arithmetic) conversion context
 *
 * Converts boolean secret sharing to arithmetic secret sharing.
 * This is essentially a MUX operation: output = choice ? 1 : 0
 */
class B2AContext {
 public:
  /**
   * @brief Construct a new B2AContext
   * @param party_id Server ID (1) or Client ID (2)
   * @param mux_context MUX context for the conversion
   */
  B2AContext(const uint32_t party_id, RotContext* rot_context,
             const uint32_t thread_num);

  ~B2AContext() = default;

  // Non-copyable
  B2AContext(const B2AContext&) = delete;
  B2AContext& operator=(const B2AContext&) = delete;

  /**
   * @brief Convert boolean secret sharing to arithmetic secret sharing
   * @param n Number of elements
   * @param input_choice Boolean secret shared input
   * @param output_z Arithmetic secret shared output
   * @param out_bw Output bit width
   */
  void B2A(uint32_t n, const uint8_t* input_choice, uint64_t* output_z,
           uint32_t out_bw);

 private:
  uint32_t party_id_;
  uint32_t thread_num_;
  std::unique_ptr<MuxContext> mux_context_;
};

}  // namespace capricorn::operator_