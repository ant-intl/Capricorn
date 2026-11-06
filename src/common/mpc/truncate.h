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

#include "src/common/mpc/b2a.h"
#include "src/common/mpc/compare.h"
#include "src/common/mpc/rot_context.h"

namespace capricorn::operator_ {

/**
 * @brief Truncation protocol context for secure right-shift operations
 *
 * Supports three truncation modes:
 * - Approximate: Efficient but with small error
 * - Positive: For positive numbers only
 * - Accurate: Uses wrap correction for exact results
 */
class TruncateContext {
 public:
  /**
   * @brief Construct a new TruncateContext
   * @param party_id Server ID (1) or Client ID (2)
   * @param rot_context ROT context for OT operations
   * @param compare_context Compare context for wrap detection
   * @param b2a_context B2A context for boolean to arithmetic conversion
   */
  TruncateContext(const uint32_t party_id, RotContext* rot_context,
                  const uint32_t thread_num);

  ~TruncateContext() = default;

  // Non-copyable
  TruncateContext(const TruncateContext&) = delete;
  TruncateContext& operator=(const TruncateContext&) = delete;

  /**
   * @brief Approximate truncation (with small error)
   * @param n Number of elements
   * @param input_x Input array
   * @param output_y Output array
   * @param bw Bit width
   * @param shift Number of bits to right-shift
   */
  void TruncateApproximate(uint32_t n, const uint64_t* input_x,
                           uint64_t* output_y, uint32_t bw, uint32_t shift);

  /**
   * @brief Positive truncation (for positive numbers only)
   * @param n Number of elements
   * @param input_x Input array (must be positive)
   * @param output_y Output array
   * @param bw Bit width
   * @param shift Number of bits to right-shift
   */
  void TruncatePositive(uint32_t n, const uint64_t* input_x, uint64_t* output_y,
                        uint32_t bw, uint32_t shift);

  /**
   * @brief Accurate truncation (uses wrap correction)
   * @param n Number of elements
   * @param input_x Input array
   * @param output_y Output array
   * @param bw Bit width
   * @param shift Number of bits to right-shift
   */
  void TruncateAccurate(uint32_t n, const uint64_t* input_x, uint64_t* output_y,
                        uint32_t bw, uint32_t shift);

  /**
   * @brief Truncate and reduce bit width
   * @param n Number of elements
   * @param input_x Input array
   * @param output_y Output array
   * @param bw Input bit width
   * @param shift Number of bits to right-shift
   * @param wrap_res Optional output for wrap result
   */
  void TruncateAndReduce(uint32_t n, const uint64_t* input_x,
                         uint64_t* output_y, uint32_t bw, uint32_t shift,
                         uint8_t* wrap_res = nullptr);

 private:
  uint32_t party_id_;
  RotContext* rot_context_;
  std::unique_ptr<CompareContext> compare_context_;
  std::unique_ptr<B2AContext> b2a_context_;
  uint32_t thread_num_;

  // Sender side implementations
  void TruncateApproximateSender(link::LinkIO* io, uint32_t len,
                                 const uint64_t* x, uint64_t* res, uint32_t bw,
                                 uint32_t shift, const emp::block* send_msg0,
                                 const emp::block* send_msg1);

  void TruncatePositiveSender(link::LinkIO* io, uint32_t len, const uint64_t* x,
                              uint64_t* res, uint32_t bw, uint32_t shift,
                              const emp::block* send_msg0,
                              const emp::block* send_msg1);

  // Receiver side implementations
  void TruncateApproximateReceiver(link::LinkIO* io, uint32_t len,
                                   const uint64_t* x, uint64_t* res,
                                   uint32_t bw, uint32_t shift,
                                   const uint8_t* recv_choice,
                                   const emp::block* recv_msg);

  void TruncatePositiveReceiver(link::LinkIO* io, uint32_t len,
                                const uint64_t* x, uint64_t* res, uint32_t bw,
                                uint32_t shift, const uint8_t* recv_choice,
                                const emp::block* recv_msg);
};

}  // namespace capricorn::operator_