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

#include "src/common/mpc/rot_context.h"
#include "src/common/runtime/link/link_io.h"

namespace capricorn::operator_ {

/**
 * @brief AND protocol context for secure AND gates
 *
 * This class implements secure AND gate protocols using ROT.
 * Supports both single-input AND (AND with local share) and
 * two-input AND (AND between two secret-shared bits).
 */
class AndContext {
 public:
  /**
   * @brief Construct a new AndContext
   * @param party_id Server ID (1) or Client ID (2)
   * @param rot_context ROT context for OT operations
   */
  AndContext(const uint32_t party_id, RotContext* rot_context,
             const uint32_t thread_num);

  ~AndContext() = default;

  // Non-copyable
  AndContext(const AndContext&) = delete;
  AndContext& operator=(const AndContext&) = delete;

  /**
   * @brief AND protocol with single input
   * output_z = input_x AND local_share
   * @param n Number of elements
   * @param input_x Input boolean array (secret shared)
   * @param output_z Output boolean array (secret shared)
   */
  void And(uint32_t n, const uint8_t* input_x, uint8_t* output_z);

  /**
   * @brief AND protocol with two inputs
   * output_z = input_x AND input_y
   * @param n Number of elements
   * @param input_x First input boolean array (secret shared)
   * @param input_y Second input boolean array (secret shared)
   * @param output_z Output boolean array (secret shared)
   */
  void And(uint32_t n, const uint8_t* input_x, const uint8_t* input_y,
           uint8_t* output_z);

  /**
   * @brief OR protocol with single input
   * output_z = input_x OR local_share
   * @param n Number of elements
   * @param input_x Input boolean array (secret shared)
   * @param output_z Output boolean array (secret shared)
   */
  void Or(uint32_t n, const uint8_t* input_x, uint8_t* output_z);

  /**
   * @brief OR protocol with two inputs
   * output_z = input_x OR input_y
   * @param n Number of elements
   * @param input_x First input boolean array (secret shared)
   * @param input_y Second input boolean array (secret shared)
   * @param output_z Output boolean array (secret shared)
   */
  void Or(uint32_t n, const uint8_t* input_x, const uint8_t* input_y,
          uint8_t* output_z);

 private:
  uint32_t party_id_;
  RotContext* rot_context_;
  uint32_t thread_num_;

  /**
   * @brief Core AND protocol implementation
   */
  void AndProtocol(link::LinkIO* io, uint32_t cur_party_id, uint32_t len,
                   const uint8_t* input_x, const uint8_t* input_y,
                   uint8_t* output_z, const emp::block* send_msg0,
                   const emp::block* send_msg1, const uint8_t* recv_choice,
                   const emp::block* recv_msg);

  /**
   * @brief AND sender side
   */
  void AndSender(link::LinkIO* io, uint32_t len, const uint8_t* input_x,
                 uint8_t* output_z, const emp::block* send_msg0,
                 const emp::block* send_msg1);

  /**
   * @brief AND receiver side
   */
  void AndReceiver(link::LinkIO* io, uint32_t len, const uint8_t* input_x,
                   uint8_t* output_z, const uint8_t* recv_choice,
                   const emp::block* recv_msg);
};

}  // namespace capricorn::operator_