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

#include "emp-tool/emp-tool.h"
#include "src/common/mpc/and_protocol.h"
#include "src/common/mpc/elementwise.h"
#include "src/common/mpc/rot_context.h"
#include "src/common/runtime/link/link_io.h"

namespace capricorn::operator_ {

/**
 * @brief Compare protocol context for secure comparison operations
 *
 * Supports:
 * - MSB extraction (sign bit)
 * - Wrap detection (carry detection in addition)
 * - Comparison operations
 */
class CompareContext {
 public:
  /**
   * @brief Construct a new CompareContext
   * @param party_id Server ID (1) or Client ID (2)
   * @param rot_context ROT context for OT operations
   * @param thread_num Number of threads for parallel processing
   */
  CompareContext(const uint32_t party_id, RotContext* rot_context,
                 const uint32_t thread_num);

  ~CompareContext() = default;

  // Non-copyable
  CompareContext(const CompareContext&) = delete;
  CompareContext& operator=(const CompareContext&) = delete;

  /**
   * @brief Extract MSB (Most Significant Bit / sign bit)
   * msb(x) = msb(x0) ^ msb(x1) ^ wrap(x0, x1)
   * @param n Number of elements
   * @param input_x Input array (secret shared)
   * @param msb_res Output MSB array (secret shared bits)
   * @param bw Bit width
   */
  void MSB(uint32_t n, const uint64_t* input_x, uint8_t* msb_res, uint32_t bw);

  /**
   * @brief Detect wrap (carry) in addition
   * wrap(x) = x + y >= 2^bw (when x and y are shares)
   * @param n Number of elements
   * @param input_x Input array (secret shared)
   * @param wrap_res Output wrap result (secret shared bits)
   * @param bw Bit width
   */
  void Wrap(uint32_t n, const uint64_t* input_x, uint8_t* wrap_res,
            uint32_t bw);

  /**
   * @brief Compare with constant value
   * @param n Number of elements
   * @param input_x Input array (secret shared)
   * @param value Constant value to compare against
   * @param cmp_res Output comparison result (secret shared bits)
   * @param bw Bit width
   * @param _less true: x < value, false: x > value
   * @param _equal Include equality in comparison
   */
  void Compare(uint32_t n, const uint64_t* input_x, uint64_t value,
               uint8_t* cmp_res, uint32_t bw, bool _less = true,
               bool _equal = false);

  /**
   * @brief Compare two secret shared values
   * @param n Number of elements
   * @param input_x First input array (secret shared)
   * @param input_y Second input array (secret shared)
   * @param cmp_res Output comparison result (secret shared bits)
   * @param bw Bit width
   * @param _less true: x < y, false: x > y
   * @param _equal Include equality in comparison
   */
  void Compare(uint32_t n, const uint64_t* input_x, const uint64_t* input_y,
               uint8_t* cmp_res, uint32_t bw, bool _less = true,
               bool _equal = false);

  /**
   * @brief Unsigned comparison
   * @param n Number of elements
   * @param input_x Input array (secret shared)
   * @param cmp_res Output comparison result (secret shared bits)
   * @param bw Bit width
   * @param _equal Include equality in comparison
   */
  void UnsignedCompare(uint32_t n, const uint64_t* input_x, uint8_t* cmp_res,
                       uint32_t bw, bool _equal = false);

 private:
  uint32_t party_id_;
  RotContext* rot_context_;
  uint32_t thread_num_;
  std::unique_ptr<AndContext> and_context_;

  // Friend declaration for testing
  friend class CompareTest_SubBlockCompareTest_Test;

  /**
   * @brief Get comparison result from sub-block comparison
   */
  void GetCompareResFromSubBlock(uint32_t n, const uint8_t* gt,
                                 const uint8_t* eq, uint8_t* res, uint32_t m);

  /**
   * @brief Sub-block comparison (mill and equal test)
   * Uses the mill_and_equal protocol from SPLUT
   */
  void SubBlockCompare(uint32_t n, const uint8_t* input_x, uint8_t* cmp_res,
                       uint8_t* equal_res, uint32_t bw);

  /**
   * @brief Mill and equal protocol sender side
   */
  void MillAndEqualSender(link::LinkIO* cur_IO, emp::block common_aes_key,
                          uint32_t len, const uint8_t* x, uint8_t* mill_res,
                          uint8_t* equal_res, uint32_t bw,
                          const emp::block* send_msg0,
                          const emp::block* send_msg1);

  /**
   * @brief Mill and equal protocol receiver side
   */
  void MillAndEqualReceiver(link::LinkIO* cur_IO, emp::block common_aes_key,
                            uint32_t len, const uint8_t* x, uint8_t* mill_res,
                            uint8_t* equal_res, uint32_t bw,
                            const uint8_t* recv_choice,
                            const emp::block* recv_msg);
};

}  // namespace capricorn::operator_