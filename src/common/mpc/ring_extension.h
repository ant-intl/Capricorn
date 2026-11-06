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

#include <sys/types.h>

#include <cstdint>
#include <memory>

#include "src/common/mpc/rot_context.h"

namespace capricorn::operator_ {

/**
 * @brief Ring extension protocol context for bit-width extension
 *
 * Extends the bit-width of secret-shared values while preserving sign.
 * This is needed when values need to be operated on with higher precision.
 */
class RingExtensionContext {
 public:
  /**
   * @brief Construct a new RingExtensionContext
   * @param party_id Server ID (1) or Client ID (2)
   * @param rot_context ROT context for OT operations
   */
  RingExtensionContext(const uint32_t party_id, RotContext* rot_context,
                       const uint32_t thread_num);

  ~RingExtensionContext() = default;

  // Non-copyable
  RingExtensionContext(const RingExtensionContext&) = delete;
  RingExtensionContext& operator=(const RingExtensionContext&) = delete;

  /**
   * @brief Extend bit-width of secret-shared values
   * @param n Number of elements
   * @param input_x Input array (in_bw bits)
   * @param output_y Output array (out_bw bits)
   * @param in_bw Input bit width
   * @param out_bw Output bit width (must be >= in_bw)
   */
  void RingExtension(uint32_t n, const uint64_t* input_x, uint64_t* output_y,
                     uint32_t in_bw, uint32_t out_bw);

  /**
   * @brief Extend values after shifting their known cleartext range positive.
   *
   * The clear values must lie in [-offset, 2^(in_bw-1) - offset). The
   * protocol adds offset in the input ring, performs the extension, and
   * subtracts it in the output ring without consuming additional ROTs.
   */
  void RingExtensionWithOffset(uint32_t n, const uint64_t* input_x,
                               uint64_t* output_y, uint32_t in_bw,
                               uint32_t out_bw, uint64_t offset);

 private:
  uint32_t party_id_;
  RotContext* rot_context_;
  uint32_t thread_num_;

  void RingExtensionSender(link::LinkIO* io, uint32_t len, const uint64_t* x,
                           uint64_t* res, uint32_t in_bw, uint32_t out_bw,
                           const emp::block* send_msg0,
                           const emp::block* send_msg1);

  void RingExtensionReceiver(link::LinkIO* io, uint32_t len, const uint64_t* x,
                             uint64_t* res, uint32_t in_bw, uint32_t out_bw,
                             const uint8_t* recv_choice,
                             const emp::block* recv_msg);
};

}  // namespace capricorn::operator_
