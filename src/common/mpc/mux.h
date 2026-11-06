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
 * @brief MUX protocol context for secure multiplexing
 *
 * This class implements secure MUX (multiplexer) protocols using ROT.
 * Supports:
 * - Single value selection: output = choice ? value : 0
 * - Shared value selection: output = choice ? shared_value : 0
 * - Two value selection: output = choice ? x : y
 */
class MuxContext {
 public:
  MuxContext(const uint32_t party_id, RotContext* rot_context,
             const uint32_t thread_num);

  ~MuxContext() = default;

  // Non-copyable
  MuxContext(const MuxContext&) = delete;
  MuxContext& operator=(const MuxContext&) = delete;

  void Mux(uint32_t n, const uint8_t* choice, uint64_t value, uint64_t* output,
           uint32_t bw);

  void Mux(uint32_t n, const uint8_t* choice, const uint64_t* x,
           uint64_t* output, uint32_t bw);

  void Mux(uint32_t n, const uint8_t* choice, const uint64_t* x,
           const uint64_t* y, uint64_t* output, uint32_t bw);

 private:
  uint32_t party_id_;
  RotContext* rot_context_;
  uint32_t thread_num_;

  /**
   * @brief Core MUX protocol implementation
   */
  void MuxProtocol(link::LinkIO* io, uint32_t cur_party_id, uint32_t len,
                   const uint8_t* choice, const uint64_t* shared_value,
                   uint64_t* output, uint32_t bw, const emp::block* send_msg0,
                   const emp::block* send_msg1, const uint8_t* recv_choice,
                   const emp::block* recv_msg);

  /**
   * @brief ROT sender side for MUX
   */
  void RotSender(link::LinkIO* io, uint32_t len, const uint8_t* choice,
                 const uint64_t* value, uint64_t* output, uint32_t bw,
                 const emp::block* send_msg0, const emp::block* send_msg1);

  /**
   * @brief ROT receiver side for MUX
   */
  void RotReceiver(link::LinkIO* io, uint32_t len, const uint8_t* choice,
                   uint64_t* output, uint32_t bw, const uint8_t* recv_choice,
                   const emp::block* recv_msg);
};

}  // namespace capricorn::operator_