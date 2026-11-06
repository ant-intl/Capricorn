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
#include <vector>

#include "emp-tool/emp-tool.h"
#include "src/common/mpc/rot_context.h"

namespace capricorn::operator_ {

/**
 * @brief Lookup table protocol context for secure table lookup
 *
 * Implements secure lookup table protocol using 1-out-of-N OT.
 * Supports single and dual table lookups.
 */
class LookupTableContext {
 public:
  /**
   * @brief Construct a new LookupTableContext
   * @param party_id Server ID (1) or Client ID (2)
   * @param rot_context ROT context for OT operations
   */
  LookupTableContext(uint32_t party_id, RotContext* rot_context,
                     const uint32_t thread_num);

  ~LookupTableContext() = default;

  // Non-copyable
  LookupTableContext(const LookupTableContext&) = delete;
  LookupTableContext& operator=(const LookupTableContext&) = delete;

  /**
   * @brief Single table lookup
   * @param n Number of elements
   * @param input_x Input index array (secret shared)
   * @param res Output result array (secret shared)
   * @param input_bw Input bit width (determines table size = 2^input_bw)
   * @param output_bw Output bit width
   * @param public_table Public table data
   */
  void LookupTable(uint32_t n, const uint8_t* input_x, uint64_t* res,
                   uint32_t input_bw, uint32_t output_bw,
                   const uint64_t* public_table);

  /**
   * @brief Dual table lookup (simultaneous lookup from two tables)
   * @param n Number of elements
   * @param input_x Input index array (secret shared)
   * @param res_x Output from first table (secret shared)
   * @param res_y Output from second table (secret shared)
   * @param input_bw Input bit width
   * @param output_bw_x First table output bit width
   * @param output_bw_y Second table output bit width
   * @param public_table_x First public table
   * @param public_table_y Second public table
   */
  void LookupTable2(uint32_t n, const uint8_t* input_x, uint64_t* res_x,
                    uint64_t* res_y, uint32_t input_bw, uint32_t output_bw_x,
                    uint32_t output_bw_y, const uint64_t* public_table_x,
                    const uint64_t* public_table_y);

 private:
  uint32_t party_id_;
  RotContext* rot_context_;
  emp::block common_aes_key_;
  uint32_t thread_num_;

  // Sender side implementations
  void LookupTableSender1(link::LinkIO* io, uint32_t len, const uint8_t* x,
                          uint64_t* res, uint32_t input_bw, uint32_t output_bw,
                          const uint64_t* public_table,
                          const emp::block* send_msg0,
                          const emp::block* send_msg1);

  void LookupTableSender2(link::LinkIO* io, uint32_t len, const uint8_t* x,
                          uint64_t* res_x, uint64_t* res_y, uint32_t input_bw,
                          uint32_t output_bw_x, uint32_t output_bw_y,
                          const uint64_t* public_table_x,
                          const uint64_t* public_table_y,
                          const emp::block* send_msg0,
                          const emp::block* send_msg1);

  // Receiver side implementations
  void LookupTableReceiver1(link::LinkIO* io, uint32_t len, const uint8_t* x,
                            uint64_t* res, uint32_t input_bw,
                            uint32_t output_bw, const uint64_t* public_table,
                            const uint8_t* recv_choice,
                            const emp::block* recv_msg);

  void LookupTableReceiver2(link::LinkIO* io, uint32_t len, const uint8_t* x,
                            uint64_t* res_x, uint64_t* res_y, uint32_t input_bw,
                            uint32_t output_bw_x, uint32_t output_bw_y,
                            const uint64_t* public_table_x,
                            const uint64_t* public_table_y,
                            const uint8_t* recv_choice,
                            const emp::block* recv_msg);
};

}  // namespace capricorn::operator_